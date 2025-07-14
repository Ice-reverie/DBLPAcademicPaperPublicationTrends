import pandas as pd
import re
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import sys
import os
import chardet
import numpy as np
import nltk
from nltk.util import ngrams

nltk.data.path.append(r"D:\anaconda\nltk_data")
nltk.download('punkt', quiet=True)

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
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "also",
    "student abstract"
}

academic_stopwords = {
    'using', 'based', 'approach', 'via', 'new', 'task',
    'image', 'detection', 'recognition', 'analysis',
    'system', 'framework', 'algorithm', 'problem', 'study',
    'performance', 'information', 'results', 'paper', 'research',
    'propose', 'proposed', 'different', 'multiple', 'two', 'one',
    'three', 'real', 'large', 'high', 'low', 'small', 'better',
    'effective', 'efficient', 'novel', 'recent', 'state', 'art',
    'stateoftheart', 'towards', 'toward', 'application',
    'applications', 'time', 'model', 'student abstract'
}
stop_words = basic_stopwords.union(academic_stopwords)

def preprocess_text(text, min_n=2, max_n=3):
    if not isinstance(text, str):
        return []

    text = re.sub(r'\s*$\s*student\s+abstract\s*$\s*', ' ', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s()]', '', text)
    text = re.sub(r'[()]', ' ', text)  # 移除括号
    text = re.sub(r'\s+', ' ', text).strip().lower()

    words = text.split()
    ngram_list = []
    for n in range(min_n, max_n + 1):
        ngram_list.extend([' '.join(gram) for gram in ngrams(words, n)])

    filtered = []
    for ngram in ngram_list:
        ngram_words = ngram.split()
        if len(ngram_words) >= 2 and all(w not in stop_words for w in ngram_words):
            filtered.append(ngram)

    return filtered

def extract_keywords(titles, top_n=30):
    all_ngrams = []
    for title in titles:
        ngrams_in_title = preprocess_text(title)
        all_ngrams.extend(ngrams_in_title)
    counter = Counter(all_ngrams)
    # 过滤停用词和 "student abstract"
    filtered_counter = Counter()
    for ngram, count in counter.items():
        ngram_lower = ngram.lower()
        if ngram_lower == "student abstract":
            continue
        valid = True
        for word in ngram.split():
            if word in stop_words:
                valid = False
                break
        if valid:
            filtered_counter[ngram] = count
    return filtered_counter.most_common(top_n)

def plot_wordcloud(keywords, conference=None, year=None):
    word_dict = dict(keywords)
    plt.figure(figsize=(12, 8))
    wc = WordCloud(
        width=1000, height=600, background_color='white',
        colormap='viridis', max_words=100, font_path='arial'
    ).generate_from_frequencies(word_dict)

    title = "Top Research Keywords"
    if conference:
        title += f" - {conference}"
    elif year:
        title += f" ({year})"
    else:
        title += " (No Conference/Year)"

    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout(pad=1.5)

    save_dir = os.path.join(os.getcwd(), "wordclouds")
    os.makedirs(save_dir, exist_ok=True)

    if conference:
        filename = f"wordcloud_{conference.replace(' ', '_')}.png"
    else:
        filename = f"wordcloud_{year}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def plot_trends(trend_data, top_n=25):
    all_keywords = list(trend_data.keys())
    if not all_keywords:
        print("没有足够关键词生成趋势图")
        return
    keyword_counter = Counter({k: sum(v.values()) for k, v in trend_data.items()})
    top_keywords = [kw for kw, _ in keyword_counter.most_common(top_n)]
    if not top_keywords:
        print("没有足够关键词生成趋势图")
        return
    all_years = sorted({year for cnt in trend_data.values() for year in cnt.keys()})
    if not all_years:
        print("没有年份数据")
        return

    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(right=0.7)
    lines = []
    labels = []

    for idx, kw in enumerate(top_keywords):
        if idx >= top_n:
            break
        cnt_year = trend_data[kw]
        trends = [cnt_year.get(year, 0) for year in all_years]
        line, = plt.plot(all_years, trends, alpha=0.7, label=kw)
        lines.append(line)
        labels.append(kw)

    plt.title('Research Keyword Trends (2020-2025)', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.legend(
        handles=lines,
        labels=labels,
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0,
        fontsize=9,
        ncol=1,
        framealpha=0.8,
        title="Top 25 Keywords",
        title_fontsize=10
    )
    plt.savefig('research_trends_top25.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("趋势图已保存: research_trends_top25.png")


def main():
    top_n = 15
    min_ngram = 2
    max_ngram = 3

    print("\n正在检测文件编码...")
    file_path = "conference_data/all_conferences_papers.csv"
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        encoding = result['encoding'] if result['confidence'] > 0.8 else 'utf-8'
        df = pd.read_csv(file_path, encoding=encoding)
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

    required_cols = {'Title', 'Year', 'Conference'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"错误：缺少必要列: {', '.join(missing_cols)}")
        sys.exit(1)

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['Year'])
    df = df[df['Year'].between(2020, 2025)]

    print("\n提取关键词中...")
    conference_keywords = defaultdict(list)
    for conference, group in df.groupby('Conference'):
        titles = group['Title'].tolist()
        keywords = extract_keywords(titles, top_n=top_n)
        conference_keywords[conference] = keywords

    print("\n生成会议主题词云...")
    for conference, keywords in conference_keywords.items():
        print(f"\n{conference} 前{top_n}关键词:")
        for i, (word, freq) in enumerate(keywords, 1):
            print(f"{i}. {word} ({freq}次)")

        plot_wordcloud(keywords, conference=conference)
        print(f"词云图已保存: {conference}_keywords.png")

    print("\n生成年度词云...")
    yearly_keywords = df.groupby('Year')['Title'].apply(extract_keywords).to_dict()
    for year, keywords in yearly_keywords.items():
        plot_wordcloud(keywords, year=year)
        print(f"词云图已保存: {year}_keywords.png")

    print("\n生成研究趋势图...")
    trend_data = defaultdict(Counter)
    for _, row in df.iterrows():
        year = row['Year']
        title = row['Title']
        words = preprocess_text(title, min_n=min_ngram, max_n=max_ngram)
        for word in words:
            trend_data[word][year] += 1

    plot_trends(trend_data, top_n=25)

    print("\n分析完成！结果已保存至当前目录")


if __name__ == "__main__":
    main()
