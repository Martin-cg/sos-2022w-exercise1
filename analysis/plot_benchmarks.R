library(ggplot2)
library(scales)
library(cowplot)
time <- 0
for (file in list.files("data/runs", full.names = T))
{

    data <- read.csv(file)
    time <- time + sum(data$runtimes)
    mplt <- ggplot(data, aes(x = Iterations, y = Markets.visited, col = as.factor(population_size))) +
        scale_x_log10(name = "Iterations per run") +
        scale_color_discrete(name = "population size") +
        scale_y_continuous(breaks=pretty_breaks()) +
        geom_jitter(width = 0.1, height = 0.2) +
        geom_smooth(span = 0.75) +
        labs(title = "GA solution quality by population size and amount of iterations", y = "markets visited")

    rplt <- ggplot(data, aes(x = Iterations, y = runtimes, col = as.factor(population_size))) +
        scale_x_log10(name = "Iterations per run") +
        scale_y_log10(name = "log runtimes/s", labels = scales::comma) +
        scale_color_discrete(name = "population size") +
        geom_jitter(width = 0.025, height = 0.05) +
        geom_smooth(span = 0.75) +
        labs(title = "GA solution time by population size and amount of iterations")

    plt <- plot_grid(mplt, rplt, nrow = 2)
    sub("[.].*", "", file)
    ggsave(paste0("data/figures/", gsub("data/runs/|_runs.csv", "", file), "_ggplot.png"), plt)

}
print(time)
