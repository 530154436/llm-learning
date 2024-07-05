### Datawhale 20224 AI夏令营
+ [基于星火大模型的群聊对话分角色要素提取挑战赛](https://challenge.xfyun.cn/topic/info?type=role-element-extraction&option=tjjg&ch=dw24_y0SCtd)
+ [星火认知大模型Web API文档](https://www.xfyun.cn/doc/spark/Web.html)
+ [第一步【零基础】跑通一站式baseline！]((https://datawhaler.feishu.cn/wiki/VIy8ws47ii2N79kOt9zcXnbXnuS))

### 上分记录
|    日期    |                 方法                 |   线上得分    | 备注                         |
|:--------:|:----------------------------------:|:---------:|:---------------------------|
| 20240630 |        提示工程(baseline.tmpl)         | 18.11970  | 官方基线，默认参数。                 |
| 20240701 |   提示工程(zero_shot.tmpl)+预处理(大模型)    |   16.3    | 修改后的prompt+大模型预处理，预处理结果不稳定 |
| 20240701 |    提示工程(zero_shot.tmpl)+预处理(正则)    | 16.82197  | 修改后的prompt+正则预处理           |
| 20240702 |    提示工程(baseline.tmpl)+预处理(正则)     | 17.21758	 | 基线prompt+正则预处理             |
| 20240702 |    提示工程(zero_shot.tmpl)+预处理(正则)    | 17.26818	 | 修改后的prompt+正则预处理           |
| 20240703 |    提示工程(zero_shot.tmpl)+预处理(正则)    | 17.86364  | 优化了一下prompt                |
| 20240704 |  微调+提示工程(zero_shot.tmpl)+预处理(正则)   | 26.58485  | 预处理+微调，微调版本(v1.0)          |
| 20240704 |  提示工程(zero_shot_v3.tmpl)+预处理(正则)   | 18.97727  | 优化prompt+正则预处理             |
| 20240704 | 微调+提示工程(zero_shot_v3.tmpl)+预处理(正则) | 26.23939  | 优化prompt+正则预处理+微调(v1.0)    |
| 20240706 | 微调+提示工程(zero_shot_v3.tmpl)+预处理(正则) | 27.07576  | 优化prompt+正则预处理+微调(v1.1)    |