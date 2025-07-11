import requests
from lxml import etree
import csv
import time
import random
import os
import pandas as pd
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 字典嵌套结构管理会议配置，只需添加新条目即可动态扩展新会议
CONFERENCES = {
    "AAAI": {
        "base_url": "https://dblp.org/db/conf/aaai/aaai{}.html",
        "start_year": 2020,
        "end_year": 2025,
        "output_file": "aaai_papers.csv"
    },
    "IJCAI": {
        "base_url": "https://dblp.org/db/conf/ijcai/ijcai{}.html",
        "start_year": 2020,
        "end_year": 2025,
        "output_file": "ijcai_papers.csv"
    },
    "CVPR": {
        "base_url": "https://dblp.org/db/conf/cvpr/cvpr{}.html",
        "start_year": 2020,
        "end_year": 2025,
        "output_file": "cvpr_papers.csv"
    },
    "ICCV": {
        "base_url": "https://dblp.org/db/conf/iccv/iccv{}.html",
        "start_year": 2020,
        "end_year": 2025,
        "output_file": "iccv_papers.csv"
    },
    "ICML": {
        "base_url": "https://dblp.org/db/conf/icml/icml{}.html",
        "start_year": 2020,
        "end_year": 2025,
        "output_file": "icml_papers.csv"
    }
}

# 会话管理模块：具有重试机制
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # 最大重试次数5
        backoff_factor=1,  # 指数退避算法
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]  # 仅对GET请求启用重试
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_random_header():
    user_agents = [  # 维护几种主流浏览器的UA
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0'
    ]
    return {  # 随机切换UA防止被识别为爬虫
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',  # 固定为英语环境
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }


def extract_paper_data(paper, year, conf_name, base_url):
    title_elem = paper.xpath('.//span[@class="title"]')
    title = title_elem[0].text.strip() if title_elem else "Title Not Available"
    authors = paper.xpath('.//span[@itemprop="author"]//span[@itemprop="name"]/text()')
    authors_str = "; ".join(authors) if authors else "Author Information Missing"
    year_elem = paper.xpath('.//span[@itemprop="datePublished"]/text()')
    pub_year = year_elem[0] if year_elem else str(year)

    # 链接提取，采用4级备选方案
    link = base_url  # 默认使用会议页面URL
    link_candidates = [
        './/nav[@class="publ"]//a[contains(@title, "digital library")]/@href',
        './/nav[@class="publ"]//a[contains(@title, "DOI")]/@href',
        './/nav[@class="publ"]//a[contains(@title, "DBLP")]/@href',
        './/a[contains(@itemprop, "url")]/@href'
    ]

    for xpath in link_candidates:
        links = paper.xpath(xpath)
        if links:
            link = urljoin(base_url, links[0])
            break

    return {  # 数据标准化管理
        'Title': title,
        'Authors': authors_str,
        'Year': pub_year,
        'Conference': f"{conf_name} {year}",
        'Conference_Type': conf_name,
        'Link': link
    }

def scrape_conference(conf_name, conf_config, session):
    base_url = conf_config["base_url"]
    output_file = conf_config["output_file"]
    start_year = conf_config["start_year"]
    end_year = conf_config["end_year"]

    # 断点续爬机制：检查文件是否存在，确定已爬取的年份
    existing_years = []
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_years = existing_df['Year'].astype(str).unique()
            print(f"已找到会议 {conf_name} 的现有数据，共包含论文 {len(existing_df)} 篇")
        except:
            existing_years = []

    # 创建或追加文件
    file_mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, file_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 如果是新文件，写入标题
        if file_mode == 'w':
            writer.writerow(['Title', 'Authors', 'Year', 'Conference', 'Conference_Type', 'Link'])

        for year in range(start_year, end_year + 1):
            if str(year) in existing_years:
                print(f"跳过 {conf_name} {year} （已经获取完毕）")
                continue

            url = base_url.format(year)
            print(f"\n从网址 {url} 中读取 {conf_name} {year} 的有关数据 ")

            try:
                response = session.get(
                    url,
                    headers=get_random_header(),
                    timeout=(15, 45)  # 连接15秒，读取45秒
                )

                if response.status_code != 200:
                    print(f"HTTP Error！状态码 {response.status_code}，年份 {year} （该年份无会议记录或网页故障）")
                    time.sleep(10)
                    continue

                response.encoding = 'utf-8'
                html = etree.HTML(response.text)

                # 查找论文元素
                papers = html.xpath('//ul[contains(@class, "publ-list")]/li[contains(@class, "entry")]')

                if not papers:
                    # 尝试备用选择器
                    papers = html.xpath('//ul[@class="publ-list"]/li')
                    print(f"使用回退选择器: 找到 {len(papers)} 篇论文")

                if not papers:
                    # 直接查找标题元素
                    papers = html.xpath('//span[@class="title"]/ancestor::li')
                    print(f"使用基于标题的选择器: 找到 {len(papers)} 篇论文")

                if not papers:
                    print(f"WARNING: {conf_name} {year} 没有论文")
                    continue

                print(f"获得会议 {conf_name} {year} 中共 {len(papers)} 篇论文")

                success_count = 0
                skipped_count = 0
                for paper in papers:
                    try:
                        paper_data = extract_paper_data(paper, year, conf_name, url)

                        if paper_data['Authors'] == "Author Information Missing":
                            print(f"缺失作者，跳过论文: {paper_data['Title']}")
                            skipped_count += 1
                            continue

                        writer.writerow([
                            paper_data['Title'],
                            paper_data['Authors'],
                            paper_data['Year'],
                            paper_data['Conference'],
                            paper_data['Conference_Type'],
                            paper_data['Link']
                        ])
                        success_count += 1
                    except Exception as e:
                        print(f"错误处理论文: {str(e)}")
                        continue

                print(f"成功保存 {conf_name} {year} 的 {success_count}/{len(papers)} 篇论文,"
                      f" 由于缺失作者跳过了 {skipped_count} 篇论文")

                # 随机延迟防止封禁
                delay = 5 + random.uniform(0, 5)
                print(f"在下一个请求之前，等待 {delay:.1f} 秒...")
                time.sleep(delay)

                # 刷新文件确保数据写入
                f.flush()

            except Exception as e:
                print(f"抓取 {conf_name} {year} 时出现严重错误: {str(e)}")
                with open('scraping_errors.log', 'a', encoding='utf-8') as log:
                    log.write(f"{conf_name} {year} 获取失败: {str(e)}\n")

                # 发生错误后延长等待时间
                time.sleep(30)

    return output_file


def main():
    print("开始爬取会议论文...")
    session = create_session()

    os.makedirs("conference_data", exist_ok=True)

    results = {}
    for conf_name, conf_config in CONFERENCES.items():
        print(f"\n{'=' * 40}")
        print(f"正在处理会议 {conf_name} 相关数据")
        print(f"{'=' * 40}")

        conf_config["output_file"] = os.path.join("conference_data", conf_config["output_file"])

        output_file = scrape_conference(conf_name, conf_config, session)
        results[conf_name] = output_file

    print("\n\n爬取总结报告:")
    print("=" * 50)
    for conf_name, file_path in results.items():
        try:
            df = pd.read_csv(file_path)
            paper_count = len(df)
            years = df['Year'].nunique()
            print(f"{conf_name}: {years} 年内共有 {paper_count} 篇论文")
        except:
            print(f"{conf_name}: 数据文件未找到或无效")

    all_files = [os.path.join("conference_data", f) for f in os.listdir("conference_data")
                 if f.endswith(".csv")]

    if all_files:
        combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        combined_file = os.path.join("conference_data", "all_conferences_papers.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"\n合并数据已保存到: {combined_file}")
        print(f"收集到的论文总数: {len(combined_df)}")

    print("\n成功完成数据爬取!")


if __name__ == "__main__":
    main()