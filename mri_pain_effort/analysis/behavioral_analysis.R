remotes::install_github("gamlj/gamlj")

library(readr)

# Behaviour
data <- read_csv("/Users/mepicard/Documents/P1_46_Behav_data_Contraction_during_Thermal_stim.csv")

data$'SubjectID' <- as.factor(data$'SubjectID')
data$'Temperature' <- as.factor(data$'Temperature')
data$'Difficulty' <- as.factor(data$'Difficulty')

GAMLj3::gamljmixed(
  formula = `Effort` ~ 1 + `Temperature` + `Difficulty` + `Temperature`:`Difficulty` + ( 1 + `Difficulty` | SubjectID ), 
  data = data
)

# Performance

perf_10 <- read_csv("/Users/mepicard/Documents/P01_46_Macro_results_10s.csv")

perf_10$'Participant' <- as.factor(perf_10$'Participant')
perf_10$'Temperature' <- as.factor(perf_10$'Temperature')
perf_10$'Difficulty' <- as.factor(perf_10$'Difficulty')

GAMLj3::gamljmixed(
  formula = COV ~ 1 + `Difficulty` + `Temperature` + `RunNumber` + `Difficulty`:`Temperature` + `Difficulty`:`RunNumber` + `Temperature`:`RunNumber` + `Difficulty`:`Temperature`:`RunNumber`+ ( 1 + `Temperature` | `Participant` ), 
  data = perf_10
)

perf_6 <- read_csv("/Users/mepicard/Documents/P01_46_Macro_results_6s.csv")

perf_6$'Participant' <- as.factor(perf_6$'Participant')
perf_6$'Temperature' <- as.factor(perf_6$'Temperature')
perf_6$'Difficulty' <- as.factor(perf_6$'Difficulty')

GAMLj3::gamljmixed(
  formula = COV ~ 1 + `Difficulty` + `Temperature` + `RunNumber` + `Difficulty`:`Temperature` + `Difficulty`:`RunNumber` + `Temperature`:`RunNumber` + `Difficulty`:`Temperature`:`RunNumber`+ ( 1 + `Temperature` | `Participant` ), 
  data = perf_6
)

perf_5 <- read_csv("/Users/mepicard/Documents/P01_46_Macro_results_5s.csv")

perf_5$'Participant' <- as.factor(perf_5$'Participant')
perf_5$'Temperature' <- as.factor(perf_5$'Temperature')
perf_5$'Difficulty' <- as.factor(perf_5$'Difficulty')

GAMLj3::gamljmixed(
  formula = COV ~ 1 + `Difficulty` + `Temperature` + `RunNumber` + `Difficulty`:`Temperature` + `Difficulty`:`RunNumber` + `Temperature`:`RunNumber` + `Difficulty`:`Temperature`:`RunNumber`+ ( 1 + `Temperature` | `Participant` ), 
  data = perf_5
)