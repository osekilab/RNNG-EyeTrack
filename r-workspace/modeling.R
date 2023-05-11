# Load packages
renv::restore()
library(lme4)
library(lmerTest)

options(scipen=10000)

# Load data
dat <- read.csv('../data/aggregated/concat/fpt-del.csv', header=TRUE)

# Scale
dat$length <- scale(dat$length)
dat$prev_length <- scale(dat$prev_length)
dat$count_ave_kika <- scale(dat$count_ave_kika)
dat$prev_cont_ave_kika <- scale(dat$prev_count_ave_kika)
dat$is_first <- as.factor(dat$is_first)
dat$is_last <- as.factor(dat$is_last)
dat$is_second_last <- as.factor(dat$is_second_last)
dat$screenN <- scale(dat$screenN)
dat$lineN <- scale(dat$lineN)
dat$segmentN <- scale(dat$segmentN)

dat$article <- as.factor(dat$article)
dat$subj <- as.factor(dat$subj)

for (seed in c('1111', '1112', '1113')){
  colName <- paste('lstm', seed, sep = '_')
  dat[, colName] <- scale(dat[, colName])
}
rm(seed, colName)

for (strategy in c('top_down', 'in_order')){
  for (seed in c('3435', '3436', '3437')){
    for (beam in c('100', '200', '400', '600', '800', '1000')){
      colName <- paste('rnng', strategy, seed, beam, sep = '_')
      dat[, colName] <- scale(dat[, colName])
    }
  }
}
rm(strategy, seed, beam, colName)

# Build baseline regression model
## Build initial baseline regression model
lmerBaseline <- lmer(logtime ~ length + prev_length + count_ave_kika + prev_cont_ave_kika + is_first + is_last + is_second_last + screenN + lineN + segmentN + (1 | subj) + (1 | article), data = dat)

## Remove data points beyond three standard deviations
datTrim <- dat[scale(resid(lmerBaseline)) < 3.0,]
lmerBaseline <- lmer(logtime ~ length + prev_length + count_ave_kika + prev_cont_ave_kika + is_first + is_last + is_second_last + screenN + lineN + segmentN + (1 | subj) + (1 | article), data = datTrim)

# Add surprisal from each LMs
## Add surprisal from LSTM
lmerLSTMs = c()
deltaDevianceLSTMs = c()
colNameLSTMs = c()
for (seed in c('1111', '1112', '1113')){
  colNameLSTM <- paste('lstm', seed, sep = '_')
  colNameLSTMs <- c(colNameLSTMs, c(colNameLSTM))

  surprisalLSTM <- datTrim[, colNameLSTM]

  lmerLSTM <- update(lmerBaseline, . ~ . + surprisalLSTM)
  deltaDevianceLSTM <- -2 * head(logLik(lmerBaseline) - logLik(lmerLSTM))

  lmerLSTMs <- c(lmerLSTMs, c(lmerLSTM))
  deltaDevianceLSTMs <- c(deltaDevianceLSTMs , deltaDevianceLSTM)
}
rm(seed, colNameLSTM, surprisalLSTM, lmerLSTM, deltaDevianceLSTM)

## Add surprisals from top-down RNNG
lmerTopDownRNNGs = c()
deltaDevianceTopDownRNNGs = c()
colNameTopDownRNNGs = c()
for (seed in c('3435', '3436', '3437')){
  for (beam in c('100', '200', '400', '600', '800', '1000')){
    colNameTopDownRNNG <- paste('rnng_top_down', seed, beam, sep = '_')
    colNameTopDownRNNGs <- c(colNameTopDownRNNGs, c(colNameTopDownRNNG))
    
    surprisalTopDownRNNG <- datTrim[, colNameTopDownRNNG]
    
    lmerTopDownRNNG <- update(lmerBaseline, . ~ . + surprisalTopDownRNNG)
    deltaDevianceTopDownRNNG <- -2 * head(logLik(lmerBaseline) - logLik(lmerTopDownRNNG))
    
    lmerTopDownRNNGs <- c(lmerTopDownRNNGs, c(lmerTopDownRNNG))
    deltaDevianceTopDownRNNGs <- c(deltaDevianceTopDownRNNGs, c(deltaDevianceTopDownRNNG))
  }
}
rm(seed, beam, colNameTopDownRNNG, surprisalTopDownRNNG, lmerTopDownRNNG, deltaDevianceTopDownRNNG)

## Add surprisals from left-corner RNNG
lmerLeftCornerRNNGs = c()
deltaDevianceLeftCornerRNNGs = c()
colNameLeftCornerRNNGs = c()
for (seed in c('3435', '3436', '3437')){
  for (beam in c('100', '200', '400', '600', '800', '1000')){
    colNameLeftCornerRNNG <- paste('rnng_in_order', seed, beam, sep = '_')
    colNameLeftCornerRNNGs <- c(colNameLeftCornerRNNGs, c(colNameLeftCornerRNNG))
    
    surprisalLeftCornerRNNG <- datTrim[, colNameLeftCornerRNNG]
    
    lmerLeftCornerRNNG <- update(lmerBaseline, . ~ . + surprisalLeftCornerRNNG)
    deltaDevianceLeftCornerRNNG <- -2 * head(logLik(lmerBaseline) - logLik(lmerLeftCornerRNNG))
    
    lmerLeftCornerRNNGs <- c(lmerLeftCornerRNNGs, c(lmerLeftCornerRNNG))
    deltaDevianceLeftCornerRNNGs <- c(deltaDevianceLeftCornerRNNGs, c(deltaDevianceLeftCornerRNNG))
  }
}
rm(seed, beam, colNameLeftCornerRNNG, surprisalLeftCornerRNNG, lmerLeftCornerRNNG, deltaDevianceLeftCornerRNNG)

# Check delta deviances
## LSTM
print(deltaDevianceLSTMs)

## Top-down RNNG
print(deltaDevianceTopDownRNNGs)

## Left-corner RNNG
print(deltaDevianceLeftCornerRNNGs)

# Extract best results
## LSTM
lmerBestLSTM <- lmerLSTMs[[which.max(deltaDevianceLSTMs)]]
colNameBestLSTM <- colNameLSTMs[[which.max(deltaDevianceLSTMs)]]

## Top-down RNNG
lmerBestTopDownRNNG <- lmerTopDownRNNGs[[which.max(deltaDevianceTopDownRNNGs)]]
colNameBestTopDownRNNG <- colNameTopDownRNNGs[[which.max(deltaDevianceTopDownRNNGs)]]

## Left-corner RNNG
lmerBestLeftCornerRNNG <- lmerLeftCornerRNNGs[[which.max(deltaDevianceLeftCornerRNNGs)]]
colNameBestLeftCornerRNNG <- colNameLeftCornerRNNGs[[which.max(deltaDevianceLeftCornerRNNGs)]]

# Nested model comparison
## Baseline < LSTM
anova(lmerBaseline, lmerBestLSTM)

## Baseline < Top-down RNNG
anova(lmerBaseline, lmerBestTopDownRNNG)

## Baseline < Left-corner RNNG
anova(lmerBaseline, lmerBestLeftCornerRNNG)

## LSTM < Top-down RNNG
lmerBestLSTMTopDownRNNG <- update(lmerBaseline, . ~ . + datTrim[, colNameBestLSTM] + datTrim[, colNameBestTopDownRNNG])
anova(lmerBestLSTM, lmerBestLSTMTopDownRNNG)

## LSTM < Left-corner RNNG
lmerBestLSTMLeftCornerRNNG <- update(lmerBaseline, . ~ . + datTrim[, colNameBestLSTM] + datTrim[, colNameBestLeftCornerRNNG])
anova(lmerBestLSTM, lmerBestLSTMLeftCornerRNNG)

## Top-down RNNG < LSTM
anova(lmerBestTopDownRNNG, lmerBestLSTMTopDownRNNG)

## Top-down RNNG < Left-corner RNNG
lmerBestTopDownRNNGLeftCornerRNNG <- update(lmerBaseline, . ~ . + datTrim[, colNameBestTopDownRNNG] + datTrim[, colNameBestLeftCornerRNNG])
anova(lmerBestTopDownRNNG, lmerBestTopDownRNNGLeftCornerRNNG)

## Left-corner RNNG < LSTM
anova(lmerBestLeftCornerRNNG, lmerBestLSTMLeftCornerRNNG)

## Left-corner RNNG < top-down RNNG
anova(lmerBestLeftCornerRNNG, lmerBestTopDownRNNGLeftCornerRNNG)

