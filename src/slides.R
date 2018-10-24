#!/usr/bin/env Rscript

library(rpart)
library(partykit)

img <- file.path("..", "img")
set.seed(999)

## boostrap some trees
n <- nrow(cu.summary)
form <- Price ~ Mileage + Type + Country

png(file.path(img, "boot_trees.png"))
opar <- par(mfrow=c(3,3), xpd=FALSE, oma=c(0,0,3,0))
for (i in 1:9) {
    inds <- sample(n, n, TRUE)
    mod <- rpart(form, data=cu.summary[inds,])
    plot(mod, uniform=TRUE)
}
mtext("Bootstrapped Regression Tress", outer=TRUE)
par(opar)
dev.off()

## class probabilities in leaf nodes example
mod <- ctree(factor(cyl) ~., data=mtcars)

png(file.path(img, "leaf_class_prob.png"))
plot(mod, terminal_panel=node_barplot, main="R mtcars: factor(cyl) ~ .")
dev.off()

## regression leaves
mod <- ctree(mpg ~ ., data=mtcars)

png(file.path(img, "leaf_reg.png"))
opar <- par(oma=c(0,0,3,0))
plot(mod, main="R mtcars: mpg ~ .")
par(opar)
dev.off()
