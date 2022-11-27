library(ggplot2)
library(scales)
library(cowplot)
time <- 0

data2 <- read.csv("data/runs/tcmt_default_runs2.csv")

plt2 <- ggplot(data2, aes(x=Iterations, y=Markets.visited, col=as.factor(Mutation.rate))) +
    facet_grid(Mutation.operator~.) +
    scale_color_discrete(name = "Mutation operator") +
    scale_y_continuous(breaks=pretty_breaks()) +
    # geom_jitter() +
    geom_smooth() +
    labs(title = "GA average solution quality with different mutation settings", y = "markets visited")

ggsave("data/figures/runs2_ggplot.png", plt2)

print(plt2)


data3 <- read.csv("data/runs/tcmt_default_runs3.csv")

plt3 <- ggplot(data3, aes(x=Iterations, y=Markets.visited, col=Selection.operator)) +
    facet_grid(Mutation.operator~.) +
    scale_color_discrete(name = "Selection Operator") +
    scale_y_continuous(breaks=pretty_breaks()) +
    # geom_jitter() +
    geom_smooth() +
    labs(title = "GA average solution quality with different selection operator", y = "markets visited")
ggsave("data/figures/runs2_ggplot.png", plt3)
print(plt3)
