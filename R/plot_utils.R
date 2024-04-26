library(argparse)
library(dplyr)
library(tidyr)
library(ggplot2)

parser <- ArgumentParser()

parser$add_argument("--plot-type", help = "Name of function to run")
parser$add_argument("--data", help = "Path to data table (tab-separated)")
parser$add_argument("--outfile", help = "Path to save plot to")

args <- parser$parse_args()

plot_aggregated_genebody_coverage <- function(data_fp, plot_fp) {
    df <- read.csv(file = data_fp, check.names = FALSE, row.names = 1)
    df$quantile <- as.integer(rownames(df))
    df <- df %>% pivot_longer(
        cols = colnames(df)[colnames(df) != "quantile"],
        names_to = "id",
        values_to = "coverage"
    )

    if (length(unique(df$id)) < 10) {
        base_plot <- ggplot(
            data = df,
            aes(x = quantile, y = coverage, color = id)
        )
    } else {
        base_plot <- ggplot(
            data = df,
            aes(x = quantile, y = coverage)
        )
    }

    plot <- base_plot +
        geom_line(aes(group = id)) +
        labs(x = "Transcript quantile (in 5' -> 3' direction)", y = "Coverage")
    print(paste("Saving plot to", getwd(), plot_fp))
    ggsave(filename = plot_fp, plot = plot)
}

if (args$plot_type == "plot_aggregated_genebody_coverage") {
    plot_aggregated_genebody_coverage(args$data, args$outfile)
}
