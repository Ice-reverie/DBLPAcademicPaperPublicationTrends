import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import sys
import chardet
import numpy as np

# 自定义英文停用词列表
basic_stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "also"
}

# 自定义学术停用词
academic_stopwords = {
    'using', 'based', 'approach', 'learning', 'via', 'network',
    'model', 'method', 'deep', 'neural', 'via', 'new', 'task',
    'data', 'image', 'detection', 'recognition', 'analysis',
    'system', 'framework', 'algorithm', 'problem', 'study', 'performance',
    'information', 'results', 'paper', 'research', 'propose', 'proposed',
    'based', 'using', 'different', 'multiple', 'two', 'one', 'three',
    'real', 'large', 'high', 'low', 'small', 'better', 'effective',
    'efficient', 'novel', 'recent', 'state', 'art', 'stateoftheart',
    'towards', 'toward', 'application', 'applications', 'time'
}

# 合并停用词集
stop_words = basic_stopwords.union(academic_stopwords)

# 文本预处理
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    # 分词并过滤停用词
    words = [word for word in text.split()
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)


def extract_keywords(titles, top_n=30):
    all_words = []
    for title in titles:
        words = title.split()
        all_words.extend(words)

    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)


def plot_wordcloud(keywords, year, conference=None):
    word_dict = dict(keywords)

    plt.figure(figsize=(12, 8))
    wc = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate_from_frequencies(word_dict)

    title = f"Top Research Keywords ({year})"
    if conference:
        title += f" - {conference}"

    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    save_dir = os.path.join(os.getcwd(), "wordcloud")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"wordcloud_{year}{'_' + conference if conference else ''}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()  # 关闭当前图像以释放内存
    print(f"词云图已保存至: {full_path}")
    plt.show()

    return filename


def plot_trends(trend_data, top_keywords=10):
    plt.figure(figsize=(14, 8))

    years = sorted(trend_data.keys())
    keywords = set()
    for year_data in trend_data.values():
        keywords.update([word for word, _ in year_data[:top_keywords]])

    # 创建趋势矩阵（数据类型为整数）
    trend_matrix = pd.DataFrame(index=sorted(keywords), columns=years, dtype=int).fillna(0)

    for year, word_list in trend_data.items():
        words = dict(word_list)
        for keyword in trend_matrix.index:
            if keyword in words:
                trend_matrix.loc[keyword, year] = words[keyword]

    top_keywords = trend_matrix.sum(axis=1).nlargest(15).index

    for keyword in top_keywords:
        plt.plot(years, trend_matrix.loc[keyword], 'o-', label=keyword, linewidth=2.5)

    plt.title('Top Research Keywords Trend (2020-2025)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(years, fontsize=12)
    plt.tight_layout()

    plt.savefig('research_trends.png', dpi=300)
    plt.show()


def main():
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "conference_data", "all_conferences_papers.csv")

    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10KB用于检测编码
            encoding_result = chardet.detect(raw_data)
            detected_encoding = encoding_result['encoding']
            confidence = encoding_result['confidence']
            print(f"检测到文件编码: {detected_encoding} (置信度: {confidence:.2%})")

            if confidence < 0.8:
                print("置信度过低，尝试常用编码...")
                try_encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'big5']
            else:
                try_encodings = [detected_encoding]

        df = None
        for encoding in try_encodings:
            try:
                print(f"尝试使用 {encoding} 编码读取文件...")
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件！")
                break
            except UnicodeDecodeError:
                print(f"使用 {encoding} 编码读取失败")
                continue
            except Exception as e:
                print(f"使用 {encoding} 编码读取时发生错误: {str(e)}")
                continue

        if df is None:
            print("所有编码尝试均失败，无法读取文件")
            sys.exit(1)

        print("\n文件前几行内容:")
        print(df.head())
        required_columns = ['title', 'year', 'conference']
        column_mapping = {
            'title': ['title', '论文标题', 'paper title', '名称', '标题', '论文名称'],
            'year': ['year', '年份', '发表年份', '年度', '出版年'],
            'conference': ['conference', '会议', '会议名称', '会议简称', '会议类型']
        }

        # 检查并重命名列
        actual_columns = df.columns.str.lower().tolist()
        found_columns = {}

        for std_col, possible_names in column_mapping.items():
            for name in possible_names:
                name_lower = name.lower()
                if name_lower in actual_columns:
                    original_col = df.columns[actual_columns.index(name_lower)]
                    print(f"找到匹配列: '{original_col}' -> '{std_col}'")
                    df.rename(columns={original_col: std_col}, inplace=True)
                    found_columns[std_col] = True
                    break

        # 检查是否所有必要列都已找到
        missing_columns = [col for col in required_columns if col not in found_columns]

        if missing_columns:
            print(f"\n错误: 数据文件中缺少必要的列: {', '.join(missing_columns)}")
            print("当前数据文件包含以下列:")
            print(df.columns.tolist())
            print("\n请检查数据文件结构并确保包含以下列:")
            print("- 论文标题 (title): 包含论文标题的列")
            print("- 发表年份 (year): 包含论文发表年份的列")
            print("- 会议名称 (conference): 包含会议名称的列")
            sys.exit(1)

        try:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        except:
            print("警告: 无法将年份列转换为整数类型，尝试使用原始数据")

        valid_years = df['year'].dropna().unique()
        df = df[df['year'].isin(valid_years)]

        print(f"\n成功读取数据！共 {len(df)} 篇论文记录。")
        print("使用的列名映射:")
        for col in required_columns:
            print(f"{col} 列的前3个值: {df[col].head(3).tolist()}")

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{file_path}'")
        print("请确保文件存在于当前目录中")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("错误: 数据文件为空")
        sys.exit(1)
    except pd.errors.ParserError:
        print("错误: 数据文件解析失败，请检查CSV格式")
        sys.exit(1)
    except Exception as e:
        print(f"读取数据时发生未知错误: {str(e)}")
        sys.exit(1)

    print("\n预处理论文标题...")
    df['clean_title'] = df['title'].apply(preprocess_text)

    yearly_keywords = {}
    print("\n年度关键词分析:")

    # 确保年份是整数类型
    years = sorted(df['year'].dropna().unique())

    for year in years:
        year_df = df[df['year'] == year]
        if len(year_df) == 0:
            continue

        keywords = extract_keywords(year_df['clean_title'])
        yearly_keywords[year] = keywords

        print(f"\n{year}年前10关键词:")
        for word, count in keywords[:10]:
            print(f"{word}: {count}")

        plot_wordcloud(keywords, year)

    conference_keywords = {}
    print("\n会议关键词分析:")

    for conference in df['conference'].unique():
        conf_df = df[df['conference'] == conference]
        keywords = extract_keywords(conf_df['clean_title'])
        conference_keywords[conference] = keywords

        print(f"\n{conference}前10关键词:")
        for word, count in keywords[:10]:
            print(f"{word}: {count}")

        plot_wordcloud(keywords, conference)

    print("\n生成研究趋势图...")
    if yearly_keywords:
        plot_trends(yearly_keywords)
    else:
        print("没有足够的数据生成趋势图")

    print("分析完成！所有图表已保存为PNG文件")


if __name__ == "__main__":
    main()