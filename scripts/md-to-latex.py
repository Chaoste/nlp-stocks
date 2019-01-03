# https://github.com/rndAdn/Markdown-latex-python3
import sys
import pandas as pd


if __name__ == "__main__":
    path = sys.argv[1]
    df = pd.read_csv(path, header=[0, 1])
    print(df.to_latex(column_format='c'))
    # md = markdown.Markdown(extensions=['latex'])
    # md_text = pandas_df_to_markdown_table(path)
    # out = md.convert(md_text)
    # print(out)
