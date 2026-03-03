# **FIRE**: A Comprehensive Benchmark for <u>**F**</u>inancial <u>**I**</u>ntelligence and <u>**R**</u>easoning <u>**E**</u>valuation.

![FIRE整体框架](fire.png)

## 1. 评测集介绍
FIRE（Finance Intelligence & Reasoning Evaluation） 是由度小满科技（北京）联合清华大学五道口金融学院与中国人民大学财政金融学院共同发布的、面向大语言模型的系统化金融能力评测基准。FIRE 在覆盖权威金融资格认证考试题目的基础上，着重检验模型在真实金融业务场景中的综合能力。在理论层面，FIRE 构建了一个包含 14,000 余道金融领域高难度、有代表性的大规模金融资格认证题库，覆盖金融学核心知识体系，并系统性地纳入安全与合规相关内容，不仅局限于对单一知识点的考察，而是从广度与深度两个维度全面评估模型对金融专业知识的掌握情况。在实践层面，FIRE 面向真实金融业务应用进行了精细化设计。我们对金融业务场景进行了科学、系统的拆解与分类，构建了一个多维度、全覆盖的评价体系，并基于真实业务实践收集了 3,000 余道场景化问题，重点考察大语言模型在金融领域的"知行合一"能力。其中，1,000 道题目具有明确的标准答案，用于评估模型在金融任务决策等方面的准确性与稳健性；其余 2,000 余道为无标准答案的开放式问题， 用于评估模型在复杂金融场景任务下的决策合理性与可解释性能力。针对这些无标准答案的问题，我们为每一道题目制定了专业、结构化的评价准则，并构建了端到端的自动化评分流水线及配套评分模型，从而实现对开放式回答的客观、可扩展评估。

相较于现有金融评测集， FIRE 通过将"标准化权威考试"与"真实业务场景评估"相结合，更贴近真实金融机构在大模型应用效果评估与风险控制方面的实际需求。

## 2. FIRE评测题

### 2.1. 金融资格认证考题集
FIRE的金融资格认证考题集是一个全面覆盖金融领域权威资格认证的评测集合，总计包含14000多道精选题目。该考题集融合了国内与国际核心从业资格认证考试题，确保对金融专业知识的全方位评估。考题覆盖14类权威的核心金融考试内容：中国人身保险从业人员资格、中国精算师、基金从业资格证、期货从业资格证、证券从业资格证、银行从业资格证、经济师（初、中、高）、AFP（金融理财师）、CFA（特许金融分析师）、CIA（国际注册内部审计师）、CISA（国际注册信息系统审计师）、CMA（注册管理会计师）、CPA（注册会计师）、FRM（金融风险管理师）。且各类型考试中，均覆盖行业金融安全合规相关内容。

在国内认证方面，深度覆盖银行/证券/基金/期货从业资格、经济师、保险从业资格、中国精算师、CPA（注册会计师）等。重点强化《证券法》、反洗钱、信贷规则等本土监管法规与合规实务，例如银行业资格中的"法律法规与综合能力"、证券业资格中的"证券市场基本法律法规"。

在国际认证方面，系统整合CFA（道德与职业准则）、FRM（风险管理与投资管理）、CIA（内部审计实务）、CISA（信息系统审计）等。着重培养遵循全球通用伦理规范与国际风险管理框架的能力，如CFA中关于利益冲突、对客户责任的经典准则。

##### 表1：FIRE 中职业资格考试来源的分类、说明与分布

<table> <thead> <tr style="background-color: transparent;"> <th width="20%">类别</th> <th width="15%">考试简称</th> <th width="65%">专业范围与说明</th> </tr> </thead> <tbody> <tr> <td rowspan="5" align="center" style="background-color: transparent;"><b>国际职业资格认证</b></td> <td align="center">CFA</td> <td>投资管理与金融分析领域的全球性权威认证。</td> </tr> <tr> <td align="center">CIA</td> <td>国际通行的内部审计专业资格标准。</td> </tr> <tr> <td align="center">CISA</td> <td>信息系统审计、控制与安全领域的全球基准认证。</td> </tr> <tr> <td align="center">CMA</td> <td>面向管理会计与财务战略的全球性职业认证。</td> </tr> <tr> <td align="center">FRM</td> <td>专注于金融风险管理与分析的国际专业认证。</td> </tr> <tr> <td rowspan="9" align="center" style="background-color: transparent;"><b>国内（中国）职业资格认证</b></td> <td align="center">AFP</td> <td>面向中国金融从业人员的金融理财师（助理级）认证。</td> </tr> <tr> <td align="center">CAA</td> <td>中国精算师国家职业资格认证。</td> </tr> <tr> <td align="center">CCBP</td> <td><b>中国银行业</b>从业人员的国家级职业资格认证。</td> </tr> <tr> <td align="center">CLIQ</td> <td>中国人身保险行业从业人员的专业资格认证。</td> </tr> <tr> <td align="center">CPA</td> <td><b>中国注册会计师</b>国家职业资格认证。</td> </tr> <tr> <td align="center">Economist</td> <td>国家经济专业技术资格考试（初级、中级、高级）。</td> </tr> <tr> <td align="center">FundPQ</td> <td>中国基金行业从业人员的法定准入资格。</td> </tr> <tr> <td align="center">FuturesPQ</td> <td>中国期货市场从业人员的入门级资格认证。</td> </tr> <tr> <td align="center">SPQ</td> <td>中国证券行业从业人员的官方资格认证。</td> </tr> </tbody> </table>




### 2.2. 真实金融场景任务集
金融领域是一个高度复杂的领域，由众多专业化子领域构成。为系统性评估大语言模型在多样化、领域特定金融业务场景中的处理能力，我们构建了一个二维评测矩阵。矩阵的行维度准确地描绘了金融业务的全景图(见表2)，确保评测任务根植于真实的业务场景，主要包括银行、保险、证券、基金管理、期货、信托、金融科技以及通用金融，并进一步细分为 17 类二级具体业务场景。矩阵的列维度提炼了贯穿所有金融业务、决定价值创造效率与质量的四大核心能力。这一划分逻辑遵循了金融机构从"洞察→创造→交付→风控"的完整价值链。在该矩阵框架指导下，我们构建了一个包含 3,000 余道高保真、场景化问题的数据集，覆盖了矩阵中大多数单元格。

##### 表2：金融行业各业务线应用场景概览

<table>
  <thead>
    <tr style="background-color: transparent;">
      <th>行业</th>
      <th>业务线</th>
      <th>洞察与决策</th>
      <th>产品与营销</th>
      <th>服务与运营</th>
      <th>风险与合规</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>银行</b></td>
      <td>公司金融</td>
      <td>财务分析、行业研究、信用评级</td>
      <td>信贷产品设计、供应链金融方案</td>
      <td>企业咨询、流程指导</td>
      <td>信用风险、反欺诈、政策合规</td>
    </tr>
    <tr>
      <td>零售金融</td>
      <td>用户画像、行为分析</td>
      <td>贷款与信用卡设计、定价、营销文案</td>
      <td>智能客服、流程指引、投诉处理</td>
      <td>个人信用风险、反欺诈、适当性管理</td>
    </tr>
    <tr>
      <td>金融市场与资金业务</td>
      <td>宏观分析、趋势预测</td>
      <td>结构化产品、交易策略</td>
      <td>交易执行支持、资金清算、报告生成</td>
      <td>市场与流动性风险、交易合规</td>
    </tr>
    <tr>
      <td rowspan="3"><b>保险</b></td>
      <td>财产保险</td>
      <td>风险因子分析、费率研究</td>
      <td>产品设计、保单条款撰写、续保方案</td>
      <td>线上理赔、咨询服务、损失评估</td>
      <td>承保风险、理赔反欺诈、责任风险</td>
    </tr>
    <tr>
      <td>人身保险</td>
      <td>健康分析、生命表研究</td>
      <td>产品设计、定价、代理人赋能</td>
      <td>智能核保、保单服务</td>
      <td>承保风险、理赔反欺诈、合规管理</td>
    </tr>
    <tr>
      <td>再保险</td>
      <td>风险暴露分析、巨灾建模</td>
      <td>再保结构设计、定价支持</td>
      <td>账单处理、对账</td>
      <td>风险评估、偿付能力、条款审查</td>
    </tr>
    <tr>
      <td rowspan="2"><b>证券</b></td>
      <td>投资银行</td>
      <td>行业研究、并购标的筛选</td>
      <td>融资方案、交易结构设计</td>
      <td>文件撰写、工作底稿</td>
      <td>尽职调查、风险识别、核验</td>
    </tr>
    <tr>
      <td>自营交易</td>
      <td>市场/个股研究、回测</td>
      <td>量化策略、研发、数据服务</td>
      <td>代码与日志生成、机构问答</td>
      <td>交易风险、策略失效、合规</td>
    </tr>
    <tr>
      <td rowspan="2"><b>基金</b></td>
      <td>私募股权</td>
      <td>BP筛选、赛道扫描、尽职调查</td>
      <td>基金结构、投资策略、募集说明书</td>
      <td>投后报告、信息披露</td>
      <td>项目风险、估值、协议审查</td>
    </tr>
    <tr>
      <td>不良资产</td>
      <td>资产估值、机会分析</td>
      <td>处置方案设计、重组方案</td>
      <td>尽调总结、流程支持</td>
      <td>处置风险、法律风险识别</td>
    </tr>
    <tr>
      <td rowspan="2"><b>期货</b></td>
      <td>经纪业务</td>
      <td>市场解读、基差分析</td>
      <td align="center">—</td>
      <td>交易规则、结算、软件指引</td>
      <td>保证金风险、强平预警、适当性</td>
    </tr>
    <tr>
      <td>风险管理</td>
      <td>基差交易、企业风险敞口分析</td>
      <td>套期保值与场外衍生品设计</td>
      <td>策略咨询、评估报告</td>
      <td>场外定价、对冲风险、合规</td>
    </tr>
    <tr>
      <td><b>信托</b></td>
      <td>投资与融资</td>
      <td>项目尽调、信用分析</td>
      <td>信托结构、交易创新</td>
      <td>存续期管理、信息披露</td>
      <td>项目风险、还款预测、合同审查</td>
    </tr>
    <tr>
      <td rowspan="2"><b>金融科技</b></td>
      <td>线上支付</td>
      <td>交易行为分析、用户画像</td>
      <td>创新产品、场景化解决方案</td>
      <td>智能客服、纠纷处理、清算</td>
      <td>反洗钱、反欺诈、支付合规</td>
    </tr>
    <tr>
      <td>技术解决方案</td>
      <td>客户需求分析、竞品分析</td>
      <td>技术方案设计、白皮书撰写</td>
      <td>技术支持、设计文档</td>
      <td>技术风险、安全与数据隐私</td>
    </tr>
    <tr>
      <td rowspan="2"><b>金融通用</b></td>
      <td>资产管理</td>
      <td>投资研究、资产配置</td>
      <td>理财产品、组合设计</td>
      <td>投后管理、绩效归因</td>
      <td>组合风险、信息披露、运营风险</td>
    </tr>
    <tr>
      <td>财富管理</td>
      <td>配置建议、趋势解读</td>
      <td>智能投顾、家族财富规划</td>
      <td>专属顾问、会议纪要</td>
      <td>客户风险、适当性、反洗钱、风险提示</td>
    </tr>
  </tbody>
</table>



### 2.3 评分方式说明
金融资格认证考题集的评估采用二元化评分目标。由于该数据集中的所有问题均为具有已验证标准答案的多项选择题，模型性能通过其预测选项与标准答案之间的一致性进行衡量。预测正确的样本记为 1 分，而任何错误或不匹配的回答均记为 0 分。

对于真实金融业务场景任务，评估方法则根据是否存在参考答案而有所区分：

- 具有参考答案的任务：
对于具备客观标准答案的场景，模型被要求以结构化 JSON 格式生成输出。我们通过自动化结果抽取流程提取最终答案，并与标准答案进行精确匹配验证。与资格类数据集的评估方式一致，匹配成功记为 1 分，未匹配成功记为 0 分。

- 开放式任务：
对于缺乏明确标准答案的任务，我们采用基于评分准则的自动化评测方法。针对不同任务构建了细粒度、任务相关的评分准则，用以刻画金融专业能力的细节。我们训练了一个专用评分模型用于理解和执行这些评分准则，该模型综合分析问题描述、问题对应的评分准则和被评测模型的输出，生成定量化的质量评分。准则生成和标注流程以及评分模型的训练请参考下图。 

![准则生成和评分模型训练](eval_model.png)

## 3. 轩辕 4.0
轩辕 4.0 是度小满科技内部"轩辕"系列金融大模型的最新旗舰版本。该模型规模为 360 亿参数，采用稠密（Dense）架构，并以 Seed-OSS-36B-Base 模型为初始化训练而成。

轩辕 4.0 的训练过程严格遵循多阶段范式。依托轩辕系列长期积累的高质量金融语料，我们首先采用了增量预训练（Continual Pre-Training, CPT）策略，将精选金融领域数据与模拟退火式训练调度相结合。为确保训练过程的稳定性与模型行为的一致性，引入了基于参考模型 KL 散度的自正则化目标函数。该机制在显著提升模型金融知识密度与推理深度的同时，有效抑制模型分布偏移，从而保障了良好的泛化能力与可持续迭代扩展性。

在完成 CPT 之后，模型进一步使用涵盖数学、STEM 以及智能体（Agent）任务的高保真数据集进行有监督微调（Supervised Fine-Tuning, SFT），以系统性增强其基础推理与问题求解能力。随后，通过可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）对模型进行精细化对齐，重点针对来源于内部真实业务场景的金融任务进行优化。该阶段采用 DAPO 算法完成策略对齐，使模型生成结果在复杂金融逻辑与业务约束下具备更高的一致性与可靠性。

## 4. 基准效果
### 4.1. 在金融资格认证考题集上的表现

<div style="overflow-x: auto;">
  <table style="border-collapse: collapse; width: 100%; font-size: 12px; text-align: center; color: #fff; border: 1px solid #fff;">
    <thead>
      <tr>
        <th rowspan="2" style="border: 1px solid #fff; padding: 10px 5px; min-width: 120px; vertical-align: middle;">考试类型</th>
        <th rowspan="2" style="border: 1px solid #fff; padding: 10px 5px; vertical-align: middle;">XuanYuan 4.0</th>
        <th colspan="5" style="border: 1px solid #fff; padding: 10px 5px; text-align: center; vertical-align: middle;">闭源模型</th>
        <th colspan="6" style="border: 1px solid #fff; padding: 10px 5px; text-align: center; vertical-align: middle;">开源模型</th>
        <th colspan="2" style="border: 1px solid #fff; padding: 10px 5px; text-align: center; vertical-align: middle;">开源金融模型</th>
      </tr>
      <tr>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Gemini-3.0</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">GPT5.2</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Claude 4.5-Sonnet</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Qwen3-Max</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Doubao-Seed-1.6</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">DeepSeek-V3.2</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Kimi K2 Thinking</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">GLM-4.6</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Qwen3-235B-A22B-Thinking-2507</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">GPT-OSS-120B</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Seed-OSS-36B</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Dianjin-R1-32B</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">XuanYuan-FinX1-Preview</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">中国人身保险从业人员资格</td><td style="border: 1px solid #fff;">89.95</td><td style="border: 1px solid #fff;">89.40</td><td style="border: 1px solid #fff;">83.95</td><td style="border: 1px solid #fff;">81.38</td><td style="border: 1px solid #fff;">85.87</td><td style="border: 1px solid #fff;">87.65</td><td style="border: 1px solid #fff;">84.67</td><td style="border: 1px solid #fff;">82.10</td><td style="border: 1px solid #fff;">81.38</td><td style="border: 1px solid #fff;">84.57</td><td style="border: 1px solid #fff;">70.78</td><td style="border: 1px solid #fff;">84.67</td><td style="border: 1px solid #fff;">84.87</td><td style="border: 1px solid #fff;">68.51</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">中国精算师</td><td style="border: 1px solid #fff;">93.17</td><td style="border: 1px solid #fff;">91.16</td><td style="border: 1px solid #fff;">94.88</td><td style="border: 1px solid #fff;">69.77</td><td style="border: 1px solid #fff;">65.89</td><td style="border: 1px solid #fff;">93.95</td><td style="border: 1px solid #fff;">95.35</td><td style="border: 1px solid #fff;">94.88</td><td style="border: 1px solid #fff;">89.77</td><td style="border: 1px solid #fff;">94.42</td><td style="border: 1px solid #fff;">91.16</td><td style="border: 1px solid #fff;">88.83</td><td style="border: 1px solid #fff;">81.86</td><td style="border: 1px solid #fff;">30.23</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">基金从业资格证</td><td style="border: 1px solid #fff;">95.51</td><td style="border: 1px solid #fff;">95.78</td><td style="border: 1px solid #fff;">87.51</td><td style="border: 1px solid #fff;">85.18</td><td style="border: 1px solid #fff;">92.66</td><td style="border: 1px solid #fff;">95.50</td><td style="border: 1px solid #fff;">91.19</td><td style="border: 1px solid #fff;">90.17</td><td style="border: 1px solid #fff;">86.70</td><td style="border: 1px solid #fff;">91.82</td><td style="border: 1px solid #fff;">75.47</td><td style="border: 1px solid #fff;">92.18</td><td style="border: 1px solid #fff;">90.29</td><td style="border: 1px solid #fff;">70.44</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">期货从业资格证</td><td style="border: 1px solid #fff;">90.46</td><td style="border: 1px solid #fff;">89.83</td><td style="border: 1px solid #fff;">79.86</td><td style="border: 1px solid #fff;">79.17</td><td style="border: 1px solid #fff;">86.15</td><td style="border: 1px solid #fff;">91.90</td><td style="border: 1px solid #fff;">84.70</td><td style="border: 1px solid #fff;">84.30</td><td style="border: 1px solid #fff;">82.13</td><td style="border: 1px solid #fff;">85.19</td><td style="border: 1px solid #fff;">70.58</td><td style="border: 1px solid #fff;">87.16</td><td style="border: 1px solid #fff;">84.10</td><td style="border: 1px solid #fff;">54.68</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">证券从业资格证</td><td style="border: 1px solid #fff;">94.72</td><td style="border: 1px solid #fff;">92.47</td><td style="border: 1px solid #fff;">82.33</td><td style="border: 1px solid #fff;">84.41</td><td style="border: 1px solid #fff;">92.22</td><td style="border: 1px solid #fff;">93.45</td><td style="border: 1px solid #fff;">90.18</td><td style="border: 1px solid #fff;">88.77</td><td style="border: 1px solid #fff;">84.41</td><td style="border: 1px solid #fff;">89.31</td><td style="border: 1px solid #fff;">70.77</td><td style="border: 1px solid #fff;">89.42</td><td style="border: 1px solid #fff;">89.53</td><td style="border: 1px solid #fff;">64.55</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">银行从业资格证</td><td style="border: 1px solid #fff;">91.61</td><td style="border: 1px solid #fff;">90.06</td><td style="border: 1px solid #fff;">82.19</td><td style="border: 1px solid #fff;">79.21</td><td style="border: 1px solid #fff;">89.04</td><td style="border: 1px solid #fff;">92.16</td><td style="border: 1px solid #fff;">85.35</td><td style="border: 1px solid #fff;">85.67</td><td style="border: 1px solid #fff;">81.28</td><td style="border: 1px solid #fff;">86.92</td><td style="border: 1px solid #fff;">68.23</td><td style="border: 1px solid #fff;">88.05</td><td style="border: 1px solid #fff;">86.54</td><td style="border: 1px solid #fff;">64.03</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">经济师 (初、中、高)</td><td style="border: 1px solid #fff;">90.47</td><td style="border: 1px solid #fff;">89.72</td><td style="border: 1px solid #fff;">83.40</td><td style="border: 1px solid #fff;">78.26</td><td style="border: 1px solid #fff;">87.44</td><td style="border: 1px solid #fff;">90.64</td><td style="border: 1px solid #fff;">85.11</td><td style="border: 1px solid #fff;">86.17</td><td style="border: 1px solid #fff;">79.84</td><td style="border: 1px solid #fff;">85.24</td><td style="border: 1px solid #fff;">66.53</td><td style="border: 1px solid #fff;">87.87</td><td style="border: 1px solid #fff;">85.11</td><td style="border: 1px solid #fff;">61.92</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">金融理财师 (AFP)</td><td style="border: 1px solid #fff;">92.36</td><td style="border: 1px solid #fff;">92.29</td><td style="border: 1px solid #fff;">86.36</td><td style="border: 1px solid #fff;">80.24</td><td style="border: 1px solid #fff;">79.84</td><td style="border: 1px solid #fff;">92.29</td><td style="border: 1px solid #fff;">88.74</td><td style="border: 1px solid #fff;">88.74</td><td style="border: 1px solid #fff;">83.79</td><td style="border: 1px solid #fff;">86.96</td><td style="border: 1px solid #fff;">77.47</td><td style="border: 1px solid #fff;">86.75</td><td style="border: 1px solid #fff;">83.20</td><td style="border: 1px solid #fff;">60.86</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">特许金融分析师 (CFA)</td><td style="border: 1px solid #fff;">92.48</td><td style="border: 1px solid #fff;">95.03</td><td style="border: 1px solid #fff;">91.65</td><td style="border: 1px solid #fff;">85.61</td><td style="border: 1px solid #fff;">83.90</td><td style="border: 1px solid #fff;">92.00</td><td style="border: 1px solid #fff;">88.28</td><td style="border: 1px solid #fff;">91.30</td><td style="border: 1px solid #fff;">87.21</td><td style="border: 1px solid #fff;">90.59</td><td style="border: 1px solid #fff;">85.79</td><td style="border: 1px solid #fff;">87.56</td><td style="border: 1px solid #fff;">79.57</td><td style="border: 1px solid #fff;">63.23</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">国际注册内部审计师 (CIA)</td><td style="border: 1px solid #fff;">91.57</td><td style="border: 1px solid #fff;">91.25</td><td style="border: 1px solid #fff;">79.84</td><td style="border: 1px solid #fff;">79.09</td><td style="border: 1px solid #fff;">82.83</td><td style="border: 1px solid #fff;">86.12</td><td style="border: 1px solid #fff;">77.57</td><td style="border: 1px solid #fff;">80.80</td><td style="border: 1px solid #fff;">74.14</td><td style="border: 1px solid #fff;">84.41</td><td style="border: 1px solid #fff;">69.96</td><td style="border: 1px solid #fff;">81.55</td><td style="border: 1px solid #fff;">73.38</td><td style="border: 1px solid #fff;">65.58</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">国际注册信息系统审计师 (CISA)</td><td style="border: 1px solid #fff;">90.85</td><td style="border: 1px solid #fff;">91.29</td><td style="border: 1px solid #fff;">82.81</td><td style="border: 1px solid #fff;">82.37</td><td style="border: 1px solid #fff;">84.00</td><td style="border: 1px solid #fff;">83.48</td><td style="border: 1px solid #fff;">74.33</td><td style="border: 1px solid #fff;">81.03</td><td style="border: 1px solid #fff;">74.55</td><td style="border: 1px solid #fff;">86.16</td><td style="border: 1px solid #fff;">70.98</td><td style="border: 1px solid #fff;">77.45</td><td style="border: 1px solid #fff;">72.99</td><td style="border: 1px solid #fff;">69.86</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">注册管理会计师 (CMA)</td><td style="border: 1px solid #fff;">95.49</td><td style="border: 1px solid #fff;">96.76</td><td style="border: 1px solid #fff;">89.90</td><td style="border: 1px solid #fff;">83.05</td><td style="border: 1px solid #fff;">83.43</td><td style="border: 1px solid #fff;">92.00</td><td style="border: 1px solid #fff;">90.10</td><td style="border: 1px solid #fff;">91.62</td><td style="border: 1px solid #fff;">89.33</td><td style="border: 1px solid #fff;">92.57</td><td style="border: 1px solid #fff;">84.19</td><td style="border: 1px solid #fff;">90.47</td><td style="border: 1px solid #fff;">84.00</td><td style="border: 1px solid #fff;">66.47</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">注册会计师 (CPA)</td><td style="border: 1px solid #fff;">91.87</td><td style="border: 1px solid #fff;">90.77</td><td style="border: 1px solid #fff;">78.24</td><td style="border: 1px solid #fff;">72.07</td><td style="border: 1px solid #fff;">82.29</td><td style="border: 1px solid #fff;">93.12</td><td style="border: 1px solid #fff;">85.69</td><td style="border: 1px solid #fff;">83.40</td><td style="border: 1px solid #fff;">78.05</td><td style="border: 1px solid #fff;">85.78</td><td style="border: 1px solid #fff;">54.86</td><td style="border: 1px solid #fff;">88.39</td><td style="border: 1px solid #fff;">82.72</td><td style="border: 1px solid #fff;">49.55</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">金融风险管理师 (FRM)</td><td style="border: 1px solid #fff;">92.57</td><td style="border: 1px solid #fff;">92.29</td><td style="border: 1px solid #fff;">91.65</td><td style="border: 1px solid #fff;">82.01</td><td style="border: 1px solid #fff;">82.80</td><td style="border: 1px solid #fff;">90.57</td><td style="border: 1px solid #fff;">88.22</td><td style="border: 1px solid #fff;">91.43</td><td style="border: 1px solid #fff;">88.87</td><td style="border: 1px solid #fff;">90.79</td><td style="border: 1px solid #fff;">85.86</td><td style="border: 1px solid #fff;">83.51</td><td style="border: 1px solid #fff;">79.22</td><td style="border: 1px solid #fff;">48.60</td></tr>
      <tr style="font-weight: bold;">
        <td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">总分</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">92.15</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">91.43</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">83.00</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">79.01</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">85.88</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">91.78</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">86.10</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">85.95</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">81.70</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">87.31</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">68.94</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">87.55</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">84.13</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">60.10</td>
      </tr>
    </tbody>
  </table>
</div>

### 4.2. 在真实金融场景任务集上的表现
<div style="overflow-x: auto;">
  <table style="border-collapse: collapse; width: 100%; font-size: 12px; text-align: center; color: #fff; border: 1px solid #fff;">
    <thead>
      <tr>
        <th rowspan="2" style="border: 1px solid #fff; padding: 10px 5px; min-width: 120px; vertical-align: middle;">行业</th>
        <th rowspan="2" style="border: 1px solid #fff; padding: 10px 5px; vertical-align: middle;">XuanYuan 4.0</th>
        <th colspan="5" style="border: 1px solid #fff; padding: 10px 5px; text-align: center; vertical-align: middle;">闭源模型</th>
        <th colspan="6" style="border: 1px solid #fff; padding: 10px 5px; text-align: center; vertical-align: middle;">开源模型</th>
        <th colspan="2" style="border: 1px solid #fff; padding: 10px 5px; text-align: center; vertical-align: middle;">开源金融模型</th>
      </tr>
      <tr>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Gemini-3.0</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">GPT5.2</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Claude 4.5-Sonnet</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Qwen3-Max</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Doubao-Seed-1.6</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">DeepSeek-V3.2</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Kimi K2 Thinking</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">GLM-4.6</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Qwen3-235B-A22B-Thinking-2507</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">GPT-OSS-120B</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Seed-OSS-36B</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">Dianjin-R1-32B</th>
        <th style="border: 1px solid #fff; padding: 10px 5px;">XuanYuan-FinX1-Preview</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">银行业 (Banking)</td><td style="border: 1px solid #fff;">78.55</td><td style="border: 1px solid #fff;">73.58</td><td style="border: 1px solid #fff;">77.41</td><td style="border: 1px solid #fff;">71.33</td><td style="border: 1px solid #fff;">71.43</td><td style="border: 1px solid #fff;">75.49</td><td style="border: 1px solid #fff;">74.87</td><td style="border: 1px solid #fff;">72.57</td><td style="border: 1px solid #fff;">72.1</td><td style="border: 1px solid #fff;">72.39</td><td style="border: 1px solid #fff;">70.12</td><td style="border: 1px solid #fff;">74.78</td><td style="border: 1px solid #fff;">62.08</td><td style="border: 1px solid #fff;">59.53</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">保险业 (Insurance)</td><td style="border: 1px solid #fff;">80.72</td><td style="border: 1px solid #fff;">79.82</td><td style="border: 1px solid #fff;">82.53</td><td style="border: 1px solid #fff;">73.42</td><td style="border: 1px solid #fff;">76.91</td><td style="border: 1px solid #fff;">79.64</td><td style="border: 1px solid #fff;">77.86</td><td style="border: 1px solid #fff;">74.85</td><td style="border: 1px solid #fff;">77.11</td><td style="border: 1px solid #fff;">76.45</td><td style="border: 1px solid #fff;">76.81</td><td style="border: 1px solid #fff;">82.79</td><td style="border: 1px solid #fff;">63.64</td><td style="border: 1px solid #fff;">73.95</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">证券业 (Securities)</td><td style="border: 1px solid #fff;">80.29</td><td style="border: 1px solid #fff;">80.62</td><td style="border: 1px solid #fff;">84.61</td><td style="border: 1px solid #fff;">77.29</td><td style="border: 1px solid #fff;">83.58</td><td style="border: 1px solid #fff;">81.27</td><td style="border: 1px solid #fff;">81.51</td><td style="border: 1px solid #fff;">80.78</td><td style="border: 1px solid #fff;">79.15</td><td style="border: 1px solid #fff;">81.1</td><td style="border: 1px solid #fff;">78.35</td><td style="border: 1px solid #fff;">81.69</td><td style="border: 1px solid #fff;">73.43</td><td style="border: 1px solid #fff;">68.4</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">基金业 (Fund Management)</td><td style="border: 1px solid #fff;">85.21</td><td style="border: 1px solid #fff;">88.73</td><td style="border: 1px solid #fff;">91.2</td><td style="border: 1px solid #fff;">85.36</td><td style="border: 1px solid #fff;">86.23</td><td style="border: 1px solid #fff;">82.74</td><td style="border: 1px solid #fff;">86.27</td><td style="border: 1px solid #fff;">86.97</td><td style="border: 1px solid #fff;">83.8</td><td style="border: 1px solid #fff;">85.21</td><td style="border: 1px solid #fff;">83.45</td><td style="border: 1px solid #fff;">87.29</td><td style="border: 1px solid #fff;">75.35</td><td style="border: 1px solid #fff;">75</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">期货业 (Futures)</td><td style="border: 1px solid #fff;">74.63</td><td style="border: 1px solid #fff;">72.79</td><td style="border: 1px solid #fff;">76.14</td><td style="border: 1px solid #fff;">70</td><td style="border: 1px solid #fff;">70.71</td><td style="border: 1px solid #fff;">73.3</td><td style="border: 1px solid #fff;">72.57</td><td style="border: 1px solid #fff;">72.22</td><td style="border: 1px solid #fff;">73.33</td><td style="border: 1px solid #fff;">71.64</td><td style="border: 1px solid #fff;">71.62</td><td style="border: 1px solid #fff;">72.67</td><td style="border: 1px solid #fff;">62.41</td><td style="border: 1px solid #fff;">62.41</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">信托业 (Trust)</td><td style="border: 1px solid #fff;">68.38</td><td style="border: 1px solid #fff;">72.06</td><td style="border: 1px solid #fff;">75</td><td style="border: 1px solid #fff;">50.74</td><td style="border: 1px solid #fff;">71.21</td><td style="border: 1px solid #fff;">65.44</td><td style="border: 1px solid #fff;">63.97</td><td style="border: 1px solid #fff;">63.97</td><td style="border: 1px solid #fff;">66.18</td><td style="border: 1px solid #fff;">66.17</td><td style="border: 1px solid #fff;">57.35</td><td style="border: 1px solid #fff;">70</td><td style="border: 1px solid #fff;">68.94</td><td style="border: 1px solid #fff;">58.09</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">互联网金融 (FinTech)</td><td style="border: 1px solid #fff;">73.17</td><td style="border: 1px solid #fff;">65.37</td><td style="border: 1px solid #fff;">60.7</td><td style="border: 1px solid #fff;">47.94</td><td style="border: 1px solid #fff;">60.06</td><td style="border: 1px solid #fff;">69.8</td><td style="border: 1px solid #fff;">64.19</td><td style="border: 1px solid #fff;">64.72</td><td style="border: 1px solid #fff;">63.53</td><td style="border: 1px solid #fff;">62.34</td><td style="border: 1px solid #fff;">61.7</td><td style="border: 1px solid #fff;">64.63</td><td style="border: 1px solid #fff;">56.44</td><td style="border: 1px solid #fff;">62.23</td></tr>
      <tr><td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">金融通用</td><td style="border: 1px solid #fff;">85.79</td><td style="border: 1px solid #fff;">83.8</td><td style="border: 1px solid #fff;">80.36</td><td style="border: 1px solid #fff;">77.73</td><td style="border: 1px solid #fff;">80.35</td><td style="border: 1px solid #fff;">85.71</td><td style="border: 1px solid #fff;">82.38</td><td style="border: 1px solid #fff;">81.75</td><td style="border: 1px solid #fff;">81.6</td><td style="border: 1px solid #fff;">84.65</td><td style="border: 1px solid #fff;">77.03</td><td style="border: 1px solid #fff;">82.29</td><td style="border: 1px solid #fff;">77.96</td><td style="border: 1px solid #fff;">73.84</td></tr>
      <tr style="font-weight: bold;">
        <td style="border: 1px solid #fff; padding: 10px 5px; text-align: left;">总分</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">79.08</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">75.72</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">77.46</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">70.22</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">73.52</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">77.15</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">75.65</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">74.19</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">73.91</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">74.35</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">71.95</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">76.34</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">65.24</td>
        <td style="border: 1px solid #fff; padding: 10px 5px;">64.9</td>
      </tr>
    </tbody>
  </table>
</div>

## 5. 评测方法
为便于后续研究与模型评测，我们已开源 FIRE—金融资格考试题集，可按照下列步骤测试。
### 5.1 安装
```bash
# recommend python >= 3.10
pip install -r requirements.txt
```
### 5.2 使用说明
```bash
# 基本启动命令
python run_evaluation.py \
  --url your_urls(可以使用多个且用空格隔开) \
  --api-key sk-your-api-key \
  --model gpt-5(部署服务的模型名) \
  --model-name gpt-5(存储结果的别名) \
  --datasets FIRE（或者FIRE_SCENE）\
  --per-url-max-workers 64 \
  --results-dir your_dir \

# resume机制：保持上述命令不变，将results-dir替换为resume-folder
python run_evaluation.py ... --resume-folder your_dir/model_results_dir

# 查看参数
python run_evaluation.py --help 
```
### 5.3 修改默认配置
数据集以及部分请求参数被配置在`config/datasets.yaml`中，例如：
```yaml
# 可以自行更改评分模型的部署信息（其中FIRE为考题集、FIRE_SCENE为真实场景）
datasets:
  FIRE_SCENE:
    name: "FIRE_SCENE"
    judge_model: "irm-32b"
    judge_model_api_type: "default"
    judge_model_api_key: "token1"
    judge_model_urls: ["your_url"]

# 支持temperature、top_p、system_prompt等参数
defaults:
  temperature: 1.0
  # top_p: 0.95
  max_tokens: 65000
  timeout: 600
  # system_prompt: ""
  extra_body: {}
```

> 注：鉴于 FIRE—真实金融场景任务集 涉及企业隐私与合规要求，相关数据暂不对外公开。

## 6. 声明

本资料库及其所包含的全部内容仅供学术研究与教育用途，不构成任何形式的金融、法律或投资建议。用户在作出任何金融、法律或投资相关决策前，应基于自身判断并主动咨询具备资质的专业人士。尽管我们力求所提供信息的准确性与完整性，但不对其准确性、完整性、时效性或适用性作出任何明示或暗示的保证。作者及相关撰稿人员不对因资料中的任何错误、遗漏，或因使用、依赖本资料库所提供的信息、软件或服务而直接或间接产生的任何损失或后果承担责任。用户因使用本资料库中的信息、软件或相关资源而产生的全部风险，均由用户自行承担。

**重要声明**：本评测数据集（FIRE）仅供学术研究和评测使用，严禁将其用于任何模型训练目的。

## 7. 问题示例
### 7.1 金融资格认证考题

---
CFA 问题

<div style="border: 1px solid #000; border-radius: 4px; padding: 1.2em; margin: 1em 0; background-color: #fff;">

**Question.**  
On 11 February 2020, the flat price and the full price of bond A are 1,014.1 and 1,026.2, respectively. Which of the following statements is *most accurate*?

<br>

**A.** Bond investors pay 1,014.1 on 11 February 2020 and 12.1 at the next coupon payment date.

**B.** Bond investors pay 1,026.2 on 11 February 2020.

**C.** Bond dealers quote 1,026.2 on 11 February 2020.

<br>

**Answer.** **B**

</div>

---
AFP 问题

<div style="border: 1px solid #000; border-radius: 4px; padding: 1.2em; margin: 1em 0; background-color: #fff;">

**Question.**  
下列关于负债比率的说法，正确的是（）。

<br>

**A.** 负债比率越低说明家庭理财效果越好。

**B.** 投资性资产可以产生收益，因此融资比率越高越好。

**C.** 流动资产作为备用金可以保证家庭的流动性，因此流动资产比例越高越好。

**D.** 如果投资收益率高于自用贷款利率，可适当提高自用贷款成数。

<br>

**Answer.** **D**

</div>

---

### 7.2 真实金融场景任务题

---
合规调查场景问题（证券业-投资银行-尽职调查）

<div style="border: 1px solid #000; border-radius: 4px; padding: 1.2em; margin: 1em 0; background-color: #fff;">

**Question.**  
你是一名合规调查员，专注于尽职调查、风险因素识别和信息披露核查。不允许在答案中添加杜撰的成分，需要基于给定的调查数据或披露文件中找到相应的信息，并进行解答。

<br>

医疗上市公司（60xxx55）2021年度报告情况……公司研发人员的数量为170，2021年度报告中披露"研发人员持续增长""研发费用资本化率大幅上升"，但对照往年年报，2018-2020历史年度报告情况……，2020年公司研发人员的数量为249，研发人员有所下降。请你核查原因，并分析研发投入真实性。

<br>

**Rubrics.**

**1 Point**  
回答完全偏离题目要求，包含大量未在提供的材料中出现的杜撰信息；或者未能基于提供的数据进行任何有效的数据对比（如未提取2020年和2021年的研发人员数量或研发投入金额）；或者完全没有遵循"合规调查员"的角色设定和分析流程，导致回答无效。

**2 Points**  
回答存在关键事实错误或逻辑矛盾，例如盲目采信2021年年报文字中"增长23.19%"的表述，而忽略了数据表中实际人数从249人下降至170人的明显矛盾；或者在计算研发投入资本化率时出现严重错误；风险判断缺失，未能识别出数据与披露描述不符的核心风险点。

**3 Points**  
回答基本完整，能够提取出2020年和2021年的关键数据（人员数量249 vs 170，资本化金额等），并计算了资本化率的变化。但在分析层面较为薄弱，未能深入探讨"披露称增长但数据下降"的矛盾原因（如未能结合"研发组织调整"或"统计口径可能变更"进行推断），或者仅罗列数据而缺乏对研发投入真实性的定性风险判断。

**4 Points**  
回答大部分符合要求，逻辑清晰。准确提取并对比了两年的人员数据（明确指出从249降至170的事实）和资本化率的激增（0.02%升至6.86%）。能够结合文本中的"研发组织调整"、"引入高端人才"等定性信息尝试解释人员变动原因，并指出了披露口径可能存在的矛盾或风险。但在语言表达的专业性（合规调查口吻）或深度风险提示上略有欠缺。

**5 Points**  
回答完美符合"合规调查员"的角色设定，严格遵循"数据分析-业务穿透-风险判断"的流程。不仅精准计算并指出了研发人员数据表面下降（249降至170）与年报文字披露（增长23.19%）之间的重大矛盾，提示了统计口径变更或披露不准确的合规风险；同时准确量化资本化率的异常大幅上升，并结合文中"3.0T MRI、CT等高端项目"的研发进展进行了合理的合理性分析。结论客观、数据详实、风险揭示到位，无任何杜撰成分。

</div>

---
