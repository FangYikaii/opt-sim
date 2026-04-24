import { computed, inject, provide, ref, type ComputedRef, type InjectionKey, type Ref } from 'vue'

export type Locale = 'zh' | 'en'

type Dictionary = Record<string, string>

const dictionaries: Record<Locale, Dictionary> = {
  zh: {
    'app.language': '中文',
    'app.languageShort': '中',
    'app.toggleLanguage': '切换到 English',
    'app.switch': '切换语言',
    'top.eyebrow': '结构色业务决策台',
    'top.currentRun': '当前方案单',
    'top.next': '下一步',
    'top.newRun': '新建需求',
    'top.reviewExport': '查看交付包',
    'top.openSummary': '查看摘要',
    'top.action.progress': '查看进度',
    'top.action.review': '进入评审',
    'top.action.export': '查看导出',
    'top.action.complete': '查看结果',
    'top.next.draft': '补全需求并开始计算',
    'top.next.simulation': '等待方案计算完成',
    'top.next.review': '确认推荐方案',
    'top.next.export': '确认导出规格',
    'top.next.complete': '查看交付结果',
    'top.next.failed': '检查失败原因并重新提交',
    'top.runId': '方案单',
    'home.eyebrow': '结构色智能设计',
    'home.title': '把颜色需求翻译成可落地的膜层方案',
    'home.body':
      '按顺序完成：检查系统状态、填写需求、评审候选方案、确认导出。',
    'home.toolbarMeta': '实验工作台',
    'home.summaryTitle': '先看这些关键信息',
    'home.summaryBody':
      '先判断系统是否可用，再看当前推荐方案、备选方案和制造提醒。训练细节与操作命令保留在下方，避免干扰业务判断。',
    'home.loading': '正在读取算法状态',
    'home.loadingNote': '正在同步训练结果、运行环境和当前可用度。',
    'home.error.overview': '算法状态读取失败',
    'home.error.run': '需求提交失败',
    'home.form.cardTitle': '发起新的设计需求',
    'home.form.cardBody':
      '用一句话描述目标外观或用途，再填写目标颜色和候选数量。',
    'home.form.requirement': '业务需求说明',
    'home.form.requirementPlaceholder':
      '例如：希望复现接近暖铜色的结构色，用于展示样片，优先保证制造稳定性和交付可行性。',
    'home.form.defaultRequirement':
      '希望复现接近暖铜色的结构色，用于展示样片，优先保证制造稳定性和交付可行性。',
    'home.form.targetHex': '目标颜色',
    'home.form.topK': '候选数量',
    'home.form.submit': '生成业务方案',
    'home.form.submitting': '正在生成方案...',
    'home.form.hint': '生成后先看推荐方案，再看制造提醒。',
    'guide.state.complete': '已完成',
    'guide.state.current': '当前',
    'guide.state.pending': '未开始',
    'guide.home.eyebrow': '实验流程',
    'guide.home.title': '按以下顺序完成实验',
    'guide.home.subtitle': '先确认系统状态，再提交需求与评审结果。',
    'guide.run.eyebrow': '实验进度',
    'guide.run.title': '当前实验步骤',
    'guide.run.subtitle': '只保留关键状态与下一步操作。',
    'workflow.viewing': '正在查看',
    'workflow.step.brief.title': '先填需求',
    'workflow.step.brief.detail': '先确认目标颜色、需求说明和基础输出约束。',
    'workflow.step.brief.action': '返回修改需求',
    'workflow.step.generate.title': '生成候选',
    'workflow.step.generate.detail': '系统执行候选生成、仿真校验和结果排序。',
    'workflow.step.generate.action': '查看生成进度',
    'workflow.step.select.title': '选择方案',
    'workflow.step.select.detail': '只比较推荐方案、备选方案和制造提醒。',
    'workflow.step.select.action': '查看候选方案',
    'workflow.step.export.title': '确认导出',
    'workflow.step.export.detail': '最后确认尺寸、格式和切片计划，再进入导出。',
    'workflow.step.export.action': '查看导出规格',
    'workflow.panel.briefLead': '这一步只确认输入条件，避免候选结果干扰当前判断。',
    'workflow.panel.selectLead': '候选已经生成完成，现在只需要比较方案并确认哪一个进入导出。',
    'workflow.panel.exportLead': '方案已确定，这一步只保留导出必需的规格和进度信息。',
    'shell.workspaceLabel': '设计工作区',
    'shell.workspaceAria': '结构色设计工作区',
    'left.targets': '目标素材',
    'left.runs': '方案记录',
    'left.contextLabel': '项目上下文',
    'left.assetType.color': '颜色目标',
    'left.assetType.image': '图像参考',
    'left.assetType.multi-view': '多视角参考',
    'timeline.brief': '需求摘要',
    'timeline.workflow': '推荐形成过程',
    'timeline.edit': '返回修改需求',
    'timeline.review': '查看交付准备',
    'timeline.statusEyebrow': '当前状态',
    'timeline.statusTitle': '实验执行面板',
    'timeline.currentStep': '当前步骤',
    'timeline.nextStep': '下一步',
    'timeline.requirement': '需求说明',
    'timeline.activity': '关键进展',
    'timeline.activityMeta': '仅展示影响决策的状态',
    'timeline.step.brief': '确认实验需求与目标参数。',
    'timeline.step.compute': '执行候选生成、仿真与排序。',
    'timeline.step.review': '比较推荐方案并确认是否进入导出。',
    'timeline.step.export': '根据导出规格生成交付结果。',
    'timeline.meta': '系统如何从需求走到候选方案',
    'timeline.targetLabel': '目标颜色',
    'timeline.heightWindow': '可制造厚度范围',
    'timeline.exportMode': '交付模式',
    'timeline.targetHex': '目标 HEX',
    'timeline.eventType.agent': '系统判断',
    'timeline.eventType.tool': '计算执行',
    'timeline.eventType.warning': '风险提醒',
    'timeline.eventType.approval': '待业务确认',
    'timeline.eventType.result': '结果输出',
    'inspector.summary': '当前推荐方案',
    'inspector.summaryMeta': '先看目标、预测成色和关键指标',
    'inspector.target': '目标颜色',
    'inspector.simulated': '预测成色',
    'inspector.processPlus': '工艺偏高情形',
    'inspector.processMinus': '工艺偏低情形',
    'inspector.keyMetrics': '业务判断要点',
    'inspector.candidates': '备选方案',
    'inspector.candidatesMeta': '只保留少量可替代方案，便于快速比较。',
    'inspector.constraints': '制造提醒',
    'inspector.constraintsMeta': '这些条件会直接影响复现实验。',
    'inspector.export': '交付准备',
    'inspector.exportMeta': '确认尺寸、格式和切片计划后再导出。',
    'inspector.select': '查看此方案',
    'inspector.selected': '当前查看',
    'inspector.algorithm': '算法准备度',
    'inspector.algorithmMeta': '用训练和评估结果判断方案可信度。',
    'inspector.rank': '排序',
    'inspector.group': '方案组',
    'inspector.metricDeltaE': '色差',
    'inspector.export.dimensions': '画幅尺寸',
    'inspector.export.fileSize': '预计文件体量',
    'inspector.export.tilePlan': '切片计划',
    'inspector.export.format': '交付格式',
    'ops.eyebrow': '操作说明',
    'ops.defaultTitle': '系统启动与运行',
    'ops.homeTitle': '系统启动步骤',
    'ops.expected': '预期结果',
    'algo.eyebrow': '算法状态',
    'algo.title': '系统状态',
    'algo.why': '流程摘要',
    'algo.current': '当前状态',
    'algo.gpu': '训练设备',
    'algo.training': '训练状态',
    'algo.environment': '运行环境',
    'algo.targets': '对比样例',
    'algo.experiments': '训练记录',
    'algo.bestOptionCount': '目标样例',
    'algo.runCount': '次记录',
    'algo.retrieval': '检索基线',
    'algo.cgan': '生成候选',
    'algo.experiment.epochs': '训练轮次',
    'algo.experiment.samples': '每色样本数',
    'algo.experiment.device': '执行设备',
    'drawer.logs': '当前进度',
    'drawer.summary': '交付摘要',
    'drawer.commands': '交付说明',
    'drawer.latest': '最新处理状态',
    'drawer.payload': '交付摘要',
    'drawer.reproduce': '交付说明',
    'drawer.notes': '交付说明',
    'drawer.logsLabel': '运行记录',
    'drawer.summaryLabel': '摘要字段',
    'drawer.commandsLabel': '调用示例',
    'drawer.log.exportTiles': '切片输出进度',
    'drawer.log.writeManifest': '交付清单写入',
    'drawer.log.approvalGate': '业务确认状态',
    'drawer.log.approved': '已确认，可进入交付准备',
    'drawer.currentStatus': '当前状态',
    'drawer.exportProgress': '导出进度',
    'drawer.tileProgress': '切片进度',
    'drawer.deliveryFormat': '交付格式',
    'drawer.deliverySize': '预计体量',
    'drawer.nextAction': '下一步',
    'common.loading': '加载中',
    'common.unavailable': '暂时不可用',
    'common.results': '个方案',
    'common.run': '方案单',
    'common.steps': '步',
    'common.targets': '个样例',
    'common.runs': '条记录',
    'common.notAvailable': '暂无',
    'common.percent': '%',
    'common.rankLabel': '第 {rank} 名',
    'status.Draft': '待整理',
    'status.Validating': '需求校验中',
    'status.Simulating': '方案仿真中',
    'status.Ranking': '结果排序中',
    'status.Running': '执行中',
    'status.Needs approval': '待业务确认',
    'status.Exporting': '交付准备中',
    'status.Complete': '已完成',
    'status.Failed': '处理失败',
    'status.ready': '已就绪',
    'status.running': '处理中',
    'status.pending': '待处理',
    'candidate.Recommended': '优先推荐',
    'candidate.Robust': '更稳妥',
    'candidate.Watch': '需留意',
    'candidate.Blocked': '暂不建议',
    'state.pass': '通过',
    'state.warning': '需关注',
    'state.fail': '不满足',
    'state.unknown': '待确认',
    'stage.Smoke run': '快速验证',
    'stage.Near paper target': '接近论文目标',
    'stage.Promising': '具备提升潜力',
    'stage.Needs full training': '需要继续训练',
    'stage.No checkpoint': '尚无模型产物',
    'stage.Metrics incomplete': '评估数据未齐',
    'winner.cgan': '生成候选更优',
    'winner.retrieval': '检索基线更优',
    'winner.tie': '效果接近',
    'metric.Best Reproduction DeltaE': '最佳复现实验色差',
    'metric.d2 Within 5 nm': '厚度恢复命中率',
    'metric.cGAN vs Retrieval': '生成候选优于检索的次数',
    'metric.Latest Training Device': '最近训练设备',
    'metric.DeltaE': '色差',
    'metric.Composite score': '综合评分',
    'metric.Process drift': '工艺波动',
    'metric.Manufacturability': '制造可行性',
    'metric.Source': '方案来源',
    'metric.High': '高',
    'metric.Medium': '中',
    'metric.Low': '低',
    'metric.CUDA': 'GPU / CUDA',
    'source.retrieval': '检索候选',
    'source.refined': '局部优化',
    'source.cGAN+TMM': '生成候选',
    'source.cGAN+refined': '生成候选 + 局部优化',
    'constraint.Ag thickness bounds': '银层厚度范围',
    'constraint.SiO2 thickness bounds': '介质层厚度范围',
    'constraint.Process sensitivity': '工艺敏感度',
    'notFound.title': '页面不存在',
    'notFound.body': '当前地址没有对应的业务页面，请返回首页重新发起需求。',
  },
  en: {
    'app.language': 'English',
    'app.languageShort': 'EN',
    'app.toggleLanguage': 'Switch to 中文',
    'app.switch': 'Switch language',
    'top.eyebrow': 'Structural Color Decision Desk',
    'top.currentRun': 'Current brief',
    'top.next': 'Next',
    'top.newRun': 'New brief',
    'top.reviewExport': 'Review deliverables',
    'top.openSummary': 'Open summary',
    'top.action.progress': 'View progress',
    'top.action.review': 'Review options',
    'top.action.export': 'View export',
    'top.action.complete': 'View results',
    'top.next.draft': 'Complete the brief and start the run',
    'top.next.simulation': 'Wait for candidate generation to finish',
    'top.next.review': 'Confirm the recommended option',
    'top.next.export': 'Confirm export specifications',
    'top.next.complete': 'Review the delivery result',
    'top.next.failed': 'Inspect the failure and submit again',
    'top.runId': 'Brief',
    'home.eyebrow': 'Structural Color Planning',
    'home.title': 'Turn color intent into a manufacturable film-stack plan',
    'home.body':
      'Follow the sequence: check system readiness, submit the brief, review options, then confirm export.',
    'home.toolbarMeta': 'Experiment workspace',
    'home.summaryTitle': 'What to review first',
    'home.summaryBody':
      'Start with system readiness, then review the recommended option, alternatives, and fabrication watchouts. Training detail and command-level instructions stay below so business users can focus on decisions.',
    'home.loading': 'Loading algorithm status',
    'home.loadingNote': 'Syncing training results, runtime environment, and current readiness.',
    'home.error.overview': 'Failed to load algorithm status',
    'home.error.run': 'Failed to submit the design brief',
    'home.form.cardTitle': 'Start a new design brief',
    'home.form.cardBody':
      'Describe the target appearance or use case in one sentence, then provide the target color and option count.',
    'home.form.requirement': 'Business requirement',
    'home.form.requirementPlaceholder':
      'For example: Reproduce a warm copper structural color for a display sample, prioritizing manufacturing stability and delivery readiness.',
    'home.form.defaultRequirement':
      'Reproduce a warm copper structural color for a display sample, prioritizing manufacturing stability and delivery readiness.',
    'home.form.targetHex': 'Target color',
    'home.form.topK': 'Options',
    'home.form.submit': 'Generate proposal',
    'home.form.submitting': 'Generating proposal...',
    'home.form.hint': 'Review the recommended option first, then check manufacturing watchouts.',
    'guide.state.complete': 'Done',
    'guide.state.current': 'Current',
    'guide.state.pending': 'Pending',
    'guide.home.eyebrow': 'Experiment flow',
    'guide.home.title': 'Complete the experiment in this order',
    'guide.home.subtitle': 'Confirm system readiness first, then submit the brief and review the result.',
    'guide.run.eyebrow': 'Experiment progress',
    'guide.run.title': 'Current experiment steps',
    'guide.run.subtitle': 'Only the key status and the next action are shown.',
    'workflow.viewing': 'Viewing',
    'workflow.step.brief.title': 'Fill brief',
    'workflow.step.brief.detail': 'Confirm the target color, requirement summary, and basic output constraints.',
    'workflow.step.brief.action': 'Edit brief',
    'workflow.step.generate.title': 'Generate options',
    'workflow.step.generate.detail': 'Run candidate generation, simulation checks, and ranking.',
    'workflow.step.generate.action': 'View generation progress',
    'workflow.step.select.title': 'Choose option',
    'workflow.step.select.detail': 'Compare only the recommended option, alternatives, and fabrication watchouts.',
    'workflow.step.select.action': 'View candidates',
    'workflow.step.export.title': 'Confirm export',
    'workflow.step.export.detail': 'Confirm dimensions, format, and tiling before export.',
    'workflow.step.export.action': 'View export specs',
    'workflow.panel.briefLead': 'This step only confirms the inputs so result panels do not distract from the brief.',
    'workflow.panel.selectLead': 'Candidates are ready. This step is only for comparing options and choosing the one to export.',
    'workflow.panel.exportLead': 'The option is fixed. This step keeps only the export-critical specs and progress.',
    'shell.workspaceLabel': 'Design workspace',
    'shell.workspaceAria': 'Structural color workspace',
    'left.targets': 'Reference inputs',
    'left.runs': 'Brief history',
    'left.contextLabel': 'Project context',
    'left.assetType.color': 'Color target',
    'left.assetType.image': 'Image reference',
    'left.assetType.multi-view': 'Multi-view reference',
    'timeline.brief': 'Brief summary',
    'timeline.workflow': 'How the recommendation was formed',
    'timeline.edit': 'Edit the brief',
    'timeline.review': 'Review deliverables',
    'timeline.statusEyebrow': 'Current status',
    'timeline.statusTitle': 'Experiment run panel',
    'timeline.currentStep': 'Current step',
    'timeline.nextStep': 'Next step',
    'timeline.requirement': 'Requirement',
    'timeline.activity': 'Key progress',
    'timeline.activityMeta': 'Only decision-relevant status is shown',
    'timeline.step.brief': 'Confirm the experiment brief and target parameters.',
    'timeline.step.compute': 'Generate, simulate, and rank candidate options.',
    'timeline.step.review': 'Compare the recommended option before export.',
    'timeline.step.export': 'Generate delivery outputs from confirmed specifications.',
    'timeline.meta': 'How the system moved from the brief to ranked options',
    'timeline.targetLabel': 'Target color',
    'timeline.heightWindow': 'Manufacturable thickness window',
    'timeline.exportMode': 'Delivery mode',
    'timeline.targetHex': 'Target HEX',
    'timeline.eventType.agent': 'System reasoning',
    'timeline.eventType.tool': 'Calculation step',
    'timeline.eventType.warning': 'Watchout',
    'timeline.eventType.approval': 'Business approval',
    'timeline.eventType.result': 'Result',
    'inspector.summary': 'Recommended option',
    'inspector.summaryMeta': 'Review target, predicted appearance, and key metrics first',
    'inspector.target': 'Target color',
    'inspector.simulated': 'Predicted appearance',
    'inspector.processPlus': 'Higher-process case',
    'inspector.processMinus': 'Lower-process case',
    'inspector.keyMetrics': 'Decision signals',
    'inspector.candidates': 'Alternative options',
    'inspector.candidatesMeta': 'Keep only a small backup set for quick comparison.',
    'inspector.constraints': 'Manufacturing watchouts',
    'inspector.constraintsMeta': 'These conditions directly affect experimental reproducibility.',
    'inspector.export': 'Delivery readiness',
    'inspector.exportMeta': 'Confirm size, format, and tiling before export.',
    'inspector.select': 'View option',
    'inspector.selected': 'Viewing',
    'inspector.algorithm': 'Algorithm readiness',
    'inspector.algorithmMeta': 'Use training and evaluation results to judge recommendation confidence.',
    'inspector.rank': 'Rank',
    'inspector.group': 'Option set',
    'inspector.metricDeltaE': 'DeltaE',
    'inspector.export.dimensions': 'Canvas size',
    'inspector.export.fileSize': 'Estimated file size',
    'inspector.export.tilePlan': 'Tiling plan',
    'inspector.export.format': 'Output format',
    'ops.eyebrow': 'Operator guide',
    'ops.defaultTitle': 'System startup and run steps',
    'ops.homeTitle': 'System startup steps',
    'ops.expected': 'Expected result',
    'algo.eyebrow': 'Algorithm status',
    'algo.title': 'System readiness',
    'algo.why': 'Flow summary',
    'algo.current': 'Current status',
    'algo.gpu': 'Training device',
    'algo.training': 'Training status',
    'algo.environment': 'Runtime environment',
    'algo.targets': 'Example comparisons',
    'algo.experiments': 'Training history',
    'algo.bestOptionCount': 'targets',
    'algo.runCount': 'records',
    'algo.retrieval': 'Retrieval baseline',
    'algo.cgan': 'Generative option',
    'algo.experiment.epochs': 'Epochs',
    'algo.experiment.samples': 'Samples per color',
    'algo.experiment.device': 'Execution device',
    'drawer.logs': 'Current progress',
    'drawer.summary': 'Delivery summary',
    'drawer.commands': 'Delivery notes',
    'drawer.latest': 'Latest processing status',
    'drawer.payload': 'Delivery summary',
    'drawer.reproduce': 'Delivery notes',
    'drawer.notes': 'Delivery notes',
    'drawer.logsLabel': 'Run log',
    'drawer.summaryLabel': 'Summary fields',
    'drawer.commandsLabel': 'Request example',
    'drawer.log.exportTiles': 'Tile export progress',
    'drawer.log.writeManifest': 'Delivery manifest',
    'drawer.log.approvalGate': 'Business approval',
    'drawer.log.approved': 'Confirmed and ready for delivery prep',
    'drawer.currentStatus': 'Current status',
    'drawer.exportProgress': 'Export progress',
    'drawer.tileProgress': 'Tile progress',
    'drawer.deliveryFormat': 'Output format',
    'drawer.deliverySize': 'Estimated size',
    'drawer.nextAction': 'Next step',
    'common.loading': 'Loading',
    'common.unavailable': 'Unavailable',
    'common.results': 'options',
    'common.run': 'Brief',
    'common.steps': 'steps',
    'common.targets': 'targets',
    'common.runs': 'records',
    'common.notAvailable': 'n/a',
    'common.percent': '%',
    'common.rankLabel': 'Rank {rank}',
    'status.Draft': 'Drafting',
    'status.Validating': 'Validating brief',
    'status.Simulating': 'Running simulation',
    'status.Ranking': 'Ranking options',
    'status.Running': 'In progress',
    'status.Needs approval': 'Pending business approval',
    'status.Exporting': 'Preparing deliverables',
    'status.Complete': 'Completed',
    'status.Failed': 'Failed',
    'status.ready': 'Ready',
    'status.running': 'Running',
    'status.pending': 'Pending',
    'candidate.Recommended': 'Recommended',
    'candidate.Robust': 'More robust',
    'candidate.Watch': 'Needs attention',
    'candidate.Blocked': 'Not recommended',
    'state.pass': 'Pass',
    'state.warning': 'Watch',
    'state.fail': 'Fail',
    'state.unknown': 'Unknown',
    'stage.Smoke run': 'Smoke validation',
    'stage.Near paper target': 'Near paper target',
    'stage.Promising': 'Promising',
    'stage.Needs full training': 'Needs full training',
    'stage.No checkpoint': 'No checkpoint yet',
    'stage.Metrics incomplete': 'Metrics incomplete',
    'winner.cgan': 'Generative option wins',
    'winner.retrieval': 'Retrieval baseline wins',
    'winner.tie': 'Comparable result',
    'metric.Best Reproduction DeltaE': 'Best reproduction DeltaE',
    'metric.d2 Within 5 nm': 'Thickness recovery hit rate',
    'metric.cGAN vs Retrieval': 'Generative wins vs retrieval',
    'metric.Latest Training Device': 'Latest training device',
    'metric.DeltaE': 'DeltaE',
    'metric.Composite score': 'Composite score',
    'metric.Process drift': 'Process drift',
    'metric.Manufacturability': 'Manufacturability',
    'metric.Source': 'Source',
    'metric.High': 'High',
    'metric.Medium': 'Medium',
    'metric.Low': 'Low',
    'metric.CUDA': 'GPU / CUDA',
    'source.retrieval': 'Retrieval seed',
    'source.refined': 'Local refinement',
    'source.cGAN+TMM': 'Generative proposal',
    'source.cGAN+refined': 'Generative + local refinement',
    'constraint.Ag thickness bounds': 'Silver thickness range',
    'constraint.SiO2 thickness bounds': 'Dielectric thickness range',
    'constraint.Process sensitivity': 'Process sensitivity',
    'notFound.title': 'Page not found',
    'notFound.body': 'There is no business page at this address. Return home and start a new brief.',
  },
}

function translate(dictionary: Dictionary, key: string): string {
  return dictionary[key] ?? key
}

function translateOrFallback(dictionary: Dictionary, key: string, fallback: string): string {
  const translated = translate(dictionary, key)
  return translated === key ? fallback : translated
}

function template(input: string, params?: Record<string, string | number>): string {
  if (!params) {
    return input
  }
  return input.replace(/\{(\w+)\}/g, (_, token: string) => String(params[token] ?? `{${token}}`))
}

export interface LocalizedLabel {
  text: string
  state: string
}

export interface I18nValue {
  locale: Ref<Locale>
  toggleLocale: () => void
  t: ComputedRef<(key: string, params?: Record<string, string | number>) => string>
  localizeCopy: ComputedRef<(input: string) => string>
  labelRunStatus: ComputedRef<(status: string) => LocalizedLabel>
  labelCandidateStatus: ComputedRef<(status: string) => LocalizedLabel>
  labelConstraintState: ComputedRef<(state: string) => LocalizedLabel>
  labelMetricState: ComputedRef<(state: string) => LocalizedLabel>
  labelStage: ComputedRef<(stage: string, stageState?: string) => LocalizedLabel>
  labelWinner: ComputedRef<(winner: string) => string>
  labelMetric: ComputedRef<(label: string) => string>
  labelMetricValue: ComputedRef<(label: string, value: string) => string>
  labelParameter: ComputedRef<(label: string) => string>
  labelSource: ComputedRef<(value: string) => string>
  labelConstraint: ComputedRef<(label: string) => string>
  labelTimelineType: ComputedRef<(type: string, fallback?: string) => string>
  labelAssetType: ComputedRef<(type: string) => string>
}

function buildStatusState(status: string): string {
  if (status === 'Needs approval') {
    return 'Needs approval'
  }
  if (status === 'Exporting' || status === 'Simulating') {
    return status
  }
  if (
    status === 'Draft' ||
    status === 'Validating' ||
    status === 'Ranking' ||
    status === 'Running' ||
    status === 'ready' ||
    status === 'running' ||
    status === 'pending'
  ) {
    return status
  }
  if (status === 'Complete') {
    return 'Complete'
  }
  if (status === 'Failed') {
    return 'Failed'
  }
  return status
}

function buildCandidateState(status: string): string {
  if (status === 'Recommended') {
    return 'Needs approval'
  }
  if (status === 'Robust') {
    return 'pass'
  }
  if (status === 'Watch') {
    return 'warning'
  }
  if (status === 'Blocked') {
    return 'fail'
  }
  return status
}

const I18N_KEY: InjectionKey<I18nValue> = Symbol('opt-sim-i18n')

export function provideI18n(): I18nValue {
  const locale = ref<Locale>('zh')
  const dictionary = computed(() => dictionaries[locale.value])

  const t = computed(() => (key: string, params?: Record<string, string | number>) => {
    return template(translate(dictionary.value, key), params)
  })

  const labelRunStatus = computed(() => (status: string): LocalizedLabel => ({
    text: translateOrFallback(dictionary.value, `status.${status}`, status),
    state: buildStatusState(status),
  }))

  const labelCandidateStatus = computed(() => (status: string): LocalizedLabel => ({
    text: translateOrFallback(dictionary.value, `candidate.${status}`, status),
    state: buildCandidateState(status),
  }))

  const labelConstraintState = computed(() => (state: string): LocalizedLabel => ({
    text: translateOrFallback(dictionary.value, `state.${state}`, state),
    state,
  }))

  const labelMetricState = computed(() => (state: string): LocalizedLabel => ({
    text: translateOrFallback(dictionary.value, `state.${state}`, state),
    state,
  }))

  const labelStage = computed(() => (stage: string, stageState = 'unknown'): LocalizedLabel => ({
    text: translateOrFallback(dictionary.value, `stage.${stage}`, stage),
    state: stageState,
  }))

  const labelWinner = computed(() => (winner: string): string => {
    if (winner === 'cgan') {
      return t.value('winner.cgan')
    }
    if (winner === 'retrieval') {
      return t.value('winner.retrieval')
    }
    return t.value('winner.tie')
  })

  const labelMetric = computed(() => (label: string): string =>
    translateOrFallback(dictionary.value, `metric.${label}`, label),
  )

  const labelSource = computed(() => (value: string): string => {
    const directKey = `source.${value}`
    const translated = translate(dictionary.value, directKey)
    return translated === directKey ? value : translated
  })

  const labelMetricValue = computed(() => (label: string, value: string): string => {
    if (label === 'Source') {
      return labelSource.value(value)
    }
    if (label === 'Manufacturability') {
      return translateOrFallback(dictionary.value, `metric.${value}`, value)
    }
    if (label === 'Latest Training Device' && value.toUpperCase() === 'CUDA') {
      return t.value('metric.CUDA')
    }
    return value
  })

  const labelParameter = computed(() => (label: string): string => {
    const mapping: Record<string, string> = {
      'Ag bottom': locale.value === 'zh' ? '底部银层' : 'Bottom Ag',
      SiO2: locale.value === 'zh' ? '二氧化硅层' : 'SiO2 layer',
      'Ag top': locale.value === 'zh' ? '顶部银层' : 'Top Ag',
    }
    return mapping[label] ?? label
  })

  const localizedConstraint = computed(() => (label: string): string =>
    translateOrFallback(dictionary.value, `constraint.${label}`, label),
  )

  const labelTimelineType = computed(() => (type: string, fallback?: string): string => {
    const key = `timeline.eventType.${type}`
    const translated = translate(dictionary.value, key)
    if (translated !== key) {
      return translated
    }
    return fallback ?? type
  })

  const labelAssetType = computed(() => (type: string): string => {
    const key = `left.assetType.${type}`
    const translated = translate(dictionary.value, key)
    return translated === key ? type : translated
  })

  const localizeCopy = computed(() => (input: string): string => {
    if (!input) {
      return input
    }
    if (locale.value === 'en') {
      const englishExact: Record<string, string> = {
        'Ag-SiO2-Ag cGAN plus thin-film inverse design': 'Ag-SiO2-Ag generative inverse design and thin-film simulation',
        'Ag-SiO2-Ag paper reproduction': 'Ag-SiO2-Ag reproduction brief',
        'Opt-Sim Microstructure Agent': 'Opt-Sim Structural Color Desk',
        'Demo target': 'Reference target',
      }
      return englishExact[input] ?? input
    }

    const zhExact: Record<string, string> = {
      'Ag-SiO2-Ag cGAN plus thin-film inverse design': 'Ag-SiO2-Ag 生成式逆向设计与薄膜仿真流程',
      'Ag-SiO2-Ag paper reproduction': 'Ag-SiO2-Ag 论文路线复现',
      'Ag-SiO2-Ag reproduction brief': 'Ag-SiO2-Ag 复现实验方案单',
      'Opt-Sim Microstructure Agent': 'Opt-Sim 结构色智能设计台',
      'Opt-Sim Structural Color Desk': 'Opt-Sim 结构色业务决策台',
      'Demo target': '示例目标',
      'Reference target': '参考目标',
      'Target color': '目标颜色',
      'Preview-first TIFF planning': '先预览、后生成 TIFF 交付',
      '512 tiles, 10k x 10k each': '共 512 个切片，每片 10k x 10k',
      '0 / 512 tiles': '已完成 0 / 512 个切片',
      '2.4 GB est.': '预计约 2.4 GB',
      '16-bit TIFF': '16-bit TIFF',
      'Paper workflow selected': '已选择论文复现路线',
      'run_inverse_design()': '执行逆向设计主流程',
      'Please confirm before adaptation/export': '请先确认后再进入交付准备',
      'paper reproduction': '论文复现',
      'retrieval + cgan + refinement': '检索 + 生成 + 局部优化',
      'awaiting review': '等待确认',
      'Best overall match after combining nominal color error with a local manufacturability refinement pass.':
        '综合色差和制造性后，该方案为当前最优。',
      'Nominal color match is usable, but small fabrication perturbations still shift color noticeably.':
        '标称条件下可用，但对工艺波动较敏感。',
      'Ag thickness bounds': '银层厚度范围',
      'SiO2 thickness bounds': '介质层厚度范围',
      'Process sensitivity': '工艺敏感度',
      'All returned Ag layers remain within the sampled 10-30 nm fabrication window.':
        '银层厚度均在 10-30 nm 可制造范围内。',
      'Dielectric thickness remains inside the 60-180 nm search window used for retrieval.':
        '介质层厚度保持在 60-180 nm 搜索范围内。',
      'Ranking includes plus/minus 0.5 nm perturbation to approximate fabrication error.':
        '排序已考虑 +/-0.5 nm 工艺扰动。',
      'Given a target structural color, generate several manufacturable Ag-SiO2-Ag thickness triples, verify them with physics simulation, and surface the most reviewable options to the user.':
        '输入目标颜色后，系统生成可制造膜层候选并给出排序结果。',
      'The cGAN proposes multiple layer-thickness candidates from a target color. Thin-film transfer-matrix simulation recalculates color error and process drift, and a local refinement pass improves both retrieval and cGAN seeds before the UI ranks candidates.':
        '流程包含候选生成、薄膜仿真、局部优化和结果排序。',
      'In plain language: the model first guesses a few film-thickness recipes for the requested color, then physics simulation checks which guesses are truly close to that color and which ones are more stable under small fabrication errors, and finally a local search nudges each promising recipe to a better nearby setting.':
        '流程：生成候选 -> 物理仿真 -> 微调优化 -> 排序输出。',
      'The full pipeline runs end-to-end, but the current cGAN quality is still far from the paper-level target. It is best understood as a validated prototype rather than a finished production model.':
        '全流程可运行，但当前模型仍处于原型验证阶段。',
      'The pipeline is close to the paper target and can be used as a strong inverse-design baseline.':
        '当前流程已接近论文指标，可作为可用基线。',
      'No cGAN experiment artifacts were found yet.': '尚未发现可用的 cGAN 训练产物。',
      'No checkpoints are available, so training cannot be considered complete.':
        '尚未检测到有效检查点。',
      'GPU usage is unknown because no experiment metrics are available.':
        '暂无指标，无法判断 GPU 训练状态。',
      'Activate the project environment': '激活项目环境',
      'Use the dedicated Conda environment so FastAPI, PyTorch, NumPy and sklearn are all available.':
        '使用项目专用的 Conda 环境，确保 FastAPI、PyTorch、NumPy 和 sklearn 依赖完整可用。',
      '`python` can import FastAPI and torch without errors.':
        '`python` 可以正常导入 FastAPI 和 torch，且无报错。',
      'Start the backend API': '启动后端接口',
      'The backend exposes inverse-design APIs, workspace data, and this algorithm overview.':
        '后端负责提供逆向设计接口、工作区数据，以及算法概况信息。',
      'Open http://127.0.0.1:8000/api/health and receive `{\"status\":\"ok\"}`.':
        '打开 http://127.0.0.1:8000/api/health，并返回 `{\"status\":\"ok\"}`。',
      'Start the frontend workspace': '启动前端工作台',
      'The Vue app shows algorithm status, candidate ranking, and the operator guide.':
        '前端页面会展示算法状态、候选方案排序和操作说明。',
      'Open http://127.0.0.1:5173 and see the home page with the algorithm overview panel.':
        '打开 http://127.0.0.1:5173，可以看到带算法概况的首页。',
      'Submit a demo business request': '提交示例业务需求',
      'Send one requirement sentence and one target color to create a reviewable run.':
        '输入一句业务需求和一个目标颜色，即可生成可评审的方案单。',
      'The response includes `activeRun`, `candidates`, `constraints`, and `exportEstimate`.':
        '返回结果会包含 `activeRun`、`candidates`、`constraints` 和 `exportEstimate`。',
      'Run a smoke re-training job': '执行快速复训',
      'Refresh the lightweight reproduction outputs quickly and update artifacts under `backend/artifacts/`.':
        '快速刷新轻量级复现实验结果，并更新 `backend/artifacts/` 下的产物。',
      '`metrics.json`, `loss_history.csv`, `candidate_samples.csv`, and `generator_checkpoint.pt` are refreshed.':
        '`metrics.json`、`loss_history.csv`、`candidate_samples.csv` 和 `generator_checkpoint.pt` 会被更新。',
      'Run a paper-grade training attempt': '执行论文级训练尝试',
      'Move from smoke validation toward the paper target metrics with many more epochs and samples.':
        '通过更高的训练轮次和样本规模，从快速验证进一步逼近论文目标指标。',
      'Compare the new `paper_reproduction` metrics against `paper_targets` after the run completes.':
        '任务完成后，将新的 `paper_reproduction` 指标与 `paper_targets` 做对比。',
      'Best current experiment is `cgan_reproduction_smoke_de2000`. Paper reference is 0.44.':
        '当前表现最好的实验是 `cgan_reproduction_smoke_de2000`，论文参考值为 0.44。',
      'Thickness recovery accuracy. Paper reference is 93.9%.':
        '厚度恢复命中率，论文参考值为 93.9%。',
      'How many demo targets were better with cGAN-generated candidates than nearest retrieval.':
        '生成候选优于检索基线的目标数量。',
    }

    if (zhExact[input]) {
      return zhExact[input]
    }

    const dynamicRules: Array<[RegExp, (...matches: string[]) => string]> = [
      [
        /^Parsed a single-target structural color request for (#[A-F0-9]+) and routed it through the Ag-SiO2-Ag paper-reproduction workflow\.$/,
        (targetHex) => `已识别目标颜色 ${targetHex}，进入论文复现流程。`,
      ],
      [
        /^Executed the hybrid inverse-design stack: retrieval seeds, cGAN proposals, thin-film simulation, and local refinement before manufacturability-aware ranking\.$/,
        () => '已完成候选生成、薄膜仿真、局部优化和排序。',
      ],
      [
        /^The reproduction-first candidate set is ready for review before broader problem adaptation\. Requirement summary: (.+)$/,
        (requirementText) => `候选方案已生成，等待确认。需求摘要：${requirementText}`,
      ],
      [
        /^(\S+) candidate remains relatively stable under \+\/-0\.5 nm process perturbation\.$/,
        (source) => `${labelSource.value(source)}在 +/-0.5 nm 扰动下保持稳定。`,
      ],
      [
        /^Training has already run and produced checkpoints in (\d+) experiment folders\. The latest artifact is `([^`]+)`\. However, the best current paper-style mean DeltaE is ([\d.]+), while the paper reference is ([\d.]+)\. Treat the model as trained, but not fully finished\.$/,
        (folderCount, latestId, bestDeltaE, paperDeltaE) =>
          `已检测到 ${folderCount} 组训练产物。最新产物为 \`${latestId}\`。当前最佳平均色差为 ${bestDeltaE}，论文参考值为 ${paperDeltaE}。`,
      ],
      [
        /^Yes\. The saved experiment artifacts show CUDA training on (.+)\.$/,
        (deviceName) => `已确认 CUDA 训练，设备为 ${deviceName}。`,
      ],
      [
        /^The code supports GPU auto-selection, but the latest saved artifact does not confirm a CUDA run\.$/,
        () => '支持自动选择 GPU，但最新产物尚未确认 CUDA 训练。',
      ],
      [
        /^Current backend environment detects CUDA and prefers GPU training on (.+)\.$/,
        (deviceName) => `运行环境已检测到 CUDA，当前优先使用 ${deviceName}。`,
      ],
      [
        /^Current backend environment falls back to CPU\. Verify the `opt_sim` environment if you expect CUDA\.$/,
        () => '运行环境当前使用 CPU。如预期应使用 CUDA，请检查 `opt_sim` 环境。',
      ],
      [
        /^(.+) ran on CUDA \((.+)\)\. Stage=(.+)\. Mean best DeltaE=([\d.]+) vs paper reference ([\d.]+)\. d2 within 5 nm=([\d.]+)%\. cGAN beat retrieval on (\d+)\/(\d+) demo targets\. Updated (.+)\.$/,
        (experimentId, deviceName, stage, deltaE, _paperDeltaE, ratio, winCount, totalCount, updatedAt) =>
          `${experimentId} 运行于 CUDA (${deviceName})。阶段：${t.value(`stage.${stage}`)}；最佳平均色差 ${deltaE}；厚度命中率 ${ratio}%；生成候选胜出 ${winCount}/${totalCount}。更新时间：${updatedAt}。`,
      ],
      [
        /^Best current experiment is `([^`]+)`\. Paper reference is ([\d.]+)\.$/,
        (experimentId, paperDeltaE) => `当前最佳实验：\`${experimentId}\`；论文参考值：${paperDeltaE}。`,
      ],
      [
        /^Group ([A-Z])$/,
        (groupName) => `方案组 ${groupName}`,
      ],
    ]

    for (const [pattern, formatter] of dynamicRules) {
      const match = input.match(pattern)
      if (match) {
        return formatter(...match.slice(1))
      }
    }

    return input
  })

  function toggleLocale(): void {
    locale.value = locale.value === 'zh' ? 'en' : 'zh'
  }

  const value: I18nValue = {
    locale,
    toggleLocale,
    t,
    localizeCopy,
    labelRunStatus,
    labelCandidateStatus,
    labelConstraintState,
    labelMetricState,
    labelStage,
    labelWinner,
    labelMetric,
    labelMetricValue,
    labelParameter,
    labelSource,
    labelConstraint: localizedConstraint,
    labelTimelineType,
    labelAssetType,
  }
  provide(I18N_KEY, value)
  return value
}

export function useI18n(): I18nValue {
  const value = inject(I18N_KEY)
  if (!value) {
    throw new Error('I18n provider is missing')
  }
  return value
}
