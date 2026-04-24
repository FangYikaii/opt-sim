from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


StatusState = Literal["pass", "warning", "fail", "unknown"]


class AlgorithmHeadlineMetric(BaseModel):
    label: str
    value: str
    detail: str
    state: StatusState = "unknown"


class AlgorithmExperiment(BaseModel):
    id: str
    updatedAt: str
    stage: str
    stageState: StatusState = "unknown"
    epochs: int | None = None
    regressorEpochs: int | None = None
    batchSize: int | None = None
    sampleCount: int | None = None
    device: str = "unknown"
    deviceName: str | None = None
    checkpointReady: bool = False
    lossHistoryReady: bool = False
    metricsReady: bool = False
    meanBestDeltaE: float | None = None
    paperTargetMeanBestDeltaE: float | None = None
    d2Within5nmRatio: float | None = None
    paperTargetD2Within5nmRatio: float | None = None
    averageRetrievalDeltaE: float | None = None
    averageCganDeltaE: float | None = None
    cganBeatsRetrievalCount: int = 0
    totalTargetsCompared: int = 0
    summary: str


class AlgorithmTargetComparison(BaseModel):
    targetHex: str
    retrievalDeltaE: float
    cganDeltaE: float
    winner: Literal["retrieval", "cgan", "tie"]


class AlgorithmOperationStep(BaseModel):
    id: str
    title: str
    description: str
    command: str | None = None
    expectedResult: str | None = None


class AlgorithmOverview(BaseModel):
    algorithmName: str
    businessGoal: str
    workflowSummary: str
    plainExplanation: str
    currentAssessment: str
    trainingConclusion: str
    gpuTrainingSummary: str
    currentEnvironmentSummary: str
    artifactRootPath: str
    guideDocumentPath: str
    bestExperimentId: str | None = None
    latestExperimentId: str | None = None
    bestExperiment: AlgorithmExperiment | None = None
    latestExperiment: AlgorithmExperiment | None = None
    experiments: list[AlgorithmExperiment]
    headlineMetrics: list[AlgorithmHeadlineMetric]
    targetComparisons: list[AlgorithmTargetComparison]
    operationSteps: list[AlgorithmOperationStep]
