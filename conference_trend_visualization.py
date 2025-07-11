import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import chardet

start_time = time.time()

current_dir = os.getcwd()
csv_file = os.path.join(current_dir, "conference_data", "all_conferences_papers.csv")

print("正在检测文件编码...")

try:
    with open(csv_file, 'rb') as f:
        raw_data = f.read(10000)  # 只阅读前10KB数据，减少IO开销
        result = chardet.detect(raw_data)  # 使用chardet检测编码
        file_encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到编码为: {file_encoding} (置信度: {confidence:.2%})")

        if confidence < 0.8:
            print("置信度低, 尝试使用常见编码...")
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(csv_file, encoding=encoding) as test_f:
                        test_f.read(100)
                    print(f"成功验证编码为 {encoding} ")
                    file_encoding = encoding
                    break
                except:
                    continue
except Exception as e:
    print(f"编码检测错误: {e}")
    file_encoding = 'utf-8'  # 默认使用utf-8
    print(f"使用默认编码: {file_encoding}")

print("\n正在读取数据...")
file_size = os.path.getsize(csv_file) / (1024 * 1024)  # 计算文件大小，以MB为单位

try:
    df = pd.read_csv(csv_file, encoding=file_encoding, on_bad_lines='warn')
    print(f"成功读取数据: 共 {len(df):,} 条记录, 文件大小: {file_size:.2f} MB")

    print("数据列名:", df.columns.tolist())

except Exception as e:
    print(f"文件读取出错: {e}")
    print("尝试 UTF-8 和 Latin1 作为编码...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print("成功使用 UTF-8 编码")
    except:
        try:
            df = pd.read_csv(csv_file, encoding='latin1')
            print("成功使用 Latin1 编码")
        except Exception as e2:
            print(f"读取失败: {e2}")
            exit()

print("\n正在进行数据预处理...")

required_columns = ['Year', 'Conference_Type']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"Error: 缺少需要的列: {', '.join(missing_cols)}")
    print("现有列:", df.columns.tolist())
    exit()

try:
    df['Year'] = df['Year'].astype(int)  # 直接转换
except Exception as e:
    print(f"年份转换出错: {e}")
    print("尝试从字符串中提取年份数字...")
    # 若失败则正则提取四位数字
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float).fillna(0).astype(int)

# 过滤目标会议
target_conferences = ['AAAI', 'IJCAI', 'CVPR', 'ICCV', 'ICML']
df = df[df['Conference_Type'].isin(target_conferences)]

# 过滤有效年份
current_year = pd.Timestamp.now().year
df = df[(df['Year'] >= 2020) & (df['Year'] <= current_year)]

print(f"过滤数据计数: {len(df):,} 条")

print("\n计算每年每个会议的论文数量...")
count_df = df.value_counts(['Conference_Type', 'Year']).reset_index(name='Count')

print("创建时间序列...")
years = list(range(2020, current_year + 1))
all_combinations = pd.DataFrame(
    [(conf, year) for conf in target_conferences for year in years],
    columns=['Conference_Type', 'Year']
)

merged_df = pd.merge(all_combinations, count_df,
                     on=['Conference_Type', 'Year'],
                     how='left').fillna(0)  # 缺失值填充为0便于后续计算
merged_df['Count'] = merged_df['Count'].astype(int)

pivot_df = merged_df.pivot(index='Year', columns='Conference_Type', values='Count')

print("\n可视化生成...")
plt.figure(figsize=(14, 8))
plt.style.use('seaborn-v0_8-whitegrid')
markers = ['o', 's', '^', 'D', 'v']
colors = ['#0077dd', '#ff7f0e', '#39c5bb', '#d62728', '#8888cc']

for i, conference in enumerate(target_conferences):
    # 筛选当前会议且Count>0的数据
    conference_data = merged_df[(merged_df['Conference_Type'] == conference) &
                                (merged_df['Count'] > 0)]

    if not conference_data.empty:
        years_with_data = conference_data['Year']
        counts = conference_data['Count']

        # 绘制趋势线
        plt.plot(years_with_data, counts,
                 marker=markers[i], markersize=8, linewidth=2.5,
                 label=conference, color=colors[i], markevery=1)

        # 添加数据标签
        for year, count in zip(pivot_df.index, pivot_df[conference]):
            if count > 0:
                # 智能标注，动态计算标签位置防止重叠
                y_offset = max(pivot_df.max().max() * 0.02, 5)
                plt.text(year, count + y_offset, f"{count:,}",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='none', pad=2))

plt.title(f'Top Conference Publication Trends (2020-{current_year})', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(pivot_df.index, fontsize=10)
plt.yticks(fontsize=10)

max_count = pivot_df.max().max()
plt.ylim(0, max_count * 1.18)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Conference', title_fontsize=12, fontsize=10,
           loc='upper left', frameon=True, shadow=True)
plt.figtext(0.5, 0.01, f"Data Source: DBLP | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            ha="center", fontsize=9, color='gray')

plt.tight_layout()
output_file = f'Conference_Paper_Trends_{current_year}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"生成图表保存为 '{output_file}'")

print("\n按会议计算年度论文数量:")
print(pivot_df.to_string())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n处理完成! 已处理 {len(df):,} 条数据")
print(f"总时间: {elapsed_time:.2f} 秒")
print(f"处理速度: 每秒 {len(df) / elapsed_time:,.0f} 条记录")

plt.show()