#################
# R code for analyzing output and plot figures
# v1.0 (organized on 2013-04-17)
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
    setwd("/Users/buxx/Desktop/test/")

#read the output file
    data <- read.table("FNV-mMMFE.txt", header=TRUE)

#calculate the statistics:

#percentage difference in expected profits of the Multi- and Single-ordering models
    data$deltaPi <- 100*(data$mean_M - data$mean_S)/data$mean_S

#percentage difference in Variance of the Multi- and Single-ordering models
    data$deltaVar <- 100*(data$var_M - data$var_S)/data$var_S

#calculate Coefficient of Variation
    data$CV_M <- sqrt(data$var_M)/data$mean_M
    data$CV_S <- sqrt(data$var_S)/data$mean_S

#percentage difference in Coefficient of Variation of the Multi- and Single-ordering models
    data$deltaCV <- 100*(data$CV_M - data$CV_S)/data$CV_S

#percentage difference in Positive Semi-Variance of the Multi- and Single-ordering models
    data$deltaSemivarU <- 100*(data$semivar_MU - data$semivar_SU)/data$semivar_SU

#percentage difference in Negative Semi-Variance of the Multi- and Single-ordering models
    data$deltaSemivarD <- 100*(data$semivar_MD - data$semivar_SD)/data$semivar_SD


#summarize the percentage difference to generate Table 1 of the paper
    summary(data[c('deltaPi', 'deltaVar', 'deltaCV', 'deltaSemivarD', 'deltaSemivarU')])



#prepare data for Figure 2
    subtotal_stdev <- aggregate(data[c('deltaPi', 'stdev')], by=list(data$stdev), FUN=mean)
    subtotal_T <- aggregate(data[c('deltaPi', 'T')], by=list(data$T), FUN=mean)
    subtotal_lambda <- aggregate(data[c('deltaPi', 'lambda')], by=list(data$lambda), FUN=mean)


#plot Figure2
    pdf('Figure2.pdf', width = 12, height = 5)
    par(oma=c(0,0,2,0))
    par(mfrow=c(1,3))


    yrange = range(subtotal_stdev$deltaPi, subtotal_T$deltaPi, subtotal_lambda$deltaPi)

    xrange = range(subtotal_stdev$stdev)
    plot(xrange, yrange, type="n", xlab=expression(sigma), ylab=expression(paste(Delta[Pi], " (%)")) , xaxt="n")
    lines(subtotal_stdev$stdev, subtotal_stdev$deltaPi, type="l")
    axis(side=1, at=seq(0.1,0.6,0.1), labels=seq(0.1,0.6,0.1))


    xrange = range(subtotal_T$T)
    plot(xrange, yrange, type="n", xlab="T", ylab="", xaxt="n")
    lines(subtotal_T$T, subtotal_T$deltaPi, type="l") 
    axis(side=1, at=seq(0.1,0.9,0.1), labels=seq(0.1,0.9,0.1))


    xrange = range(subtotal_lambda$lambda)
    plot(xrange, yrange, type="n", xlab=expression(lambda), ylab="", xaxt="n")
    lines(subtotal_lambda$lambda, subtotal_lambda$deltaPi, type="l") 
    axis(side=1, at=seq(0.02,0.2,0.02), labels=seq(0.02,0.2,0.02))

    title(main="Figure 2. Comparison of the Expected Profits Between the Multiordering and the Static Single-Ordering Strategies: m-MMFE", outer=T)

    dev.off()
