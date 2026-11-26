# 基于 LSTM 的影评情感分析系统

10 月归档的网站项目教学版。

## Python 环境与运行

~~~powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train.py --model bilstm_attention --output-dir outputs/main
python app.py --host 127.0.0.1 --port 5000
~~~

公开版保留应用源码和可公开的结构文件。导入教学数据时请使用合成记录或有权公开的数据。

## 数据说明

由项目生成的缓存和分析结果放在本地。需要外部数据或模型时，按运行入口的输入说明准备你可以使用的数据，再生成结果。
