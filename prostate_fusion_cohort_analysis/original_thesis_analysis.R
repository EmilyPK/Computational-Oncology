# ============================================================
# FINAL VERSION - TCGA links / SUC rechts
# SUC-Y-Achse: 0-15 | TCGA-Y-Achse: 10-25
# p-Werte als Signifikanzsterne; Mittelwert und Median markiert
# ============================================================

# ============================================================
# 0. Workspace und Libraries
# ============================================================

rm(list = ls())

library(ggplot2)
library(gridExtra)
library(grid)

# ============================================================
# 1. Generelle Einstellungen
# ============================================================

gene_names <- c(
  "ERG",
  "RAI2",
  "CTBP1",
  "CTBP2"
)

group_levels <- c(
  "no structural variant",
  "TMPRSS2-ERG"
)

group_colors <- c(
  "no structural variant" = "#A6CEE3",
  "TMPRSS2-ERG" = "#F4A6A6"
)

# Feste Breite der separaten Spalte mit den Gennamen.
gene_label_width <- unit(0.85, "in")

# Feste Hoehen fuer Spaltenueberschriften, Symbollegende und Schutzrand.
header_height <- unit(60, "pt")
header_top_padding <- unit(8, "pt")

# Einheitliche Darstellung der Lageparameter in allen Panels.
summary_marker_size <- 2.8
summary_marker_stroke <- 0.8

# Die Spannweite jeder Y-Achse ist halb so gross wie zuvor (15 statt 30).
# Eine halbierte Panelhoehe erhaelt daher die bisherige Hoehe je 5er-Schritt.
panel_height <- unit(249.27 / 2, "pt")

# ============================================================
# 2. SUC-Daten einlesen und vorbereiten
# ============================================================

mrna_suc <- read.delim(
  "mRNA expression new.txt",
  sep = "\t",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

clinical_suc_full <- read.delim(
  "prad_su2c_2019_clinical_data.tsv",
  sep = "\t",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

sv_suc_full <- read.delim(
  "structural_variants.txt",
  sep = "\t",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

clinical_suc <- clinical_suc_full[, c(
  "Sample ID",
  "ETS Fusion Details",
  "ETS Fusion SEQ"
)]

sv_suc <- sv_suc_full[, c(
  "SAMPLE_ID",
  "ERG"
)]

merged_suc_1 <- merge(
  mrna_suc,
  clinical_suc,
  by.x = "SAMPLE_ID",
  by.y = "Sample ID",
  all.x = TRUE
)

suc <- merge(
  merged_suc_1,
  sv_suc,
  by = "SAMPLE_ID",
  all.x = TRUE
)

genes_suc <- c(
  "ERG.x",
  "RAI2",
  "CTBP1",
  "CTBP2"
)

# Urspruengliche SUC-Filterung beibehalten.
suc <- suc[
  !(
    is.na(suc$ERG.x) |
      is.na(suc$RAI2) |
      is.na(suc$CTBP1) |
      is.na(suc$CTBP2) |
      is.na(suc$`ETS Fusion SEQ`)
  ),
  ,
  drop = FALSE
]

# Log2(Expression + 1) wie im urspruenglichen SUC-Skript.
for (g in genes_suc) {
  x <- as.character(suc[[g]])

  x[
    is.na(x) |
      x == "" |
      x == "NA"
  ] <- NA_character_

  x <- gsub("\\.", "", x)

  x[
    !grepl("^\\d+$", x)
  ] <- NA_character_

  suc[[paste0("log2_", g)]] <-
    log2(as.numeric(x) + 1)
}

genes_log2_suc <- c(
  "log2_ERG.x",
  "log2_RAI2",
  "log2_CTBP1",
  "log2_CTBP2"
)

suc <- suc[
  complete.cases(suc[, genes_log2_suc]),
  ,
  drop = FALSE
]

# Gruppierung wie im urspruenglichen SUC-Skript.
suc$Group <- ifelse(
  !is.na(suc$`ETS Fusion Details`) &
    suc$`ETS Fusion Details` == "TMPRSS2-ERG",
  "TMPRSS2-ERG",
  "no structural variant"
)

suc$Group <- factor(
  suc$Group,
  levels = group_levels
)

# Die tatsaechlich verwendeten Log2-Werte.
suc$log2_ERG <- log2(
  as.numeric(suc$ERG.x) + 1
)

suc$log2_RAI2 <- log2(
  as.numeric(suc$RAI2) + 1
)

suc$log2_CTBP1 <- log2(
  as.numeric(suc$CTBP1) + 1
)

suc$log2_CTBP2 <- log2(
  as.numeric(suc$CTBP2) + 1
)

genes_log2_suc <- c(
  "log2_ERG",
  "log2_RAI2",
  "log2_CTBP1",
  "log2_CTBP2"
)

# ============================================================
# 3. TCGA-Daten einlesen und vorbereiten
# ============================================================

mrna_tcga <- read.delim(
  "mRNA Expression TCGA.txt",
  sep = "\t",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

sv_tcga <- read.delim(
  "structural_variants TCGA.txt",
  sep = "\t",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

mrna_tcga <- mrna_tcga[, c(
  "SAMPLE_ID",
  "ERG",
  "RAI2",
  "CTBP1",
  "CTBP2"
)]

sv_tcga <- sv_tcga[, c(
  "SAMPLE_ID",
  "ERG"
)]

tcga <- merge(
  mrna_tcga,
  sv_tcga,
  by = "SAMPLE_ID",
  all.x = TRUE
)

genes_tcga <- c(
  "ERG.x",
  "RAI2",
  "CTBP1",
  "CTBP2"
)

# Log2(Expression + 1) wie im urspruenglichen TCGA-Skript.
for (gene in genes_tcga) {
  values <- as.character(tcga[[gene]])

  values[
    is.na(values) |
      values == "" |
      values == "NA"
  ] <- NA_character_

  values <- gsub("\\.", "", values)

  values[
    !grepl("^\\d+$", values)
  ] <- NA_character_

  tcga[[paste0("log2_", gene)]] <-
    log2(as.numeric(values) + 1)
}

genes_log2_tcga <- c(
  "log2_ERG.x",
  "log2_RAI2",
  "log2_CTBP1",
  "log2_CTBP2"
)

tcga <- tcga[
  complete.cases(tcga[, genes_log2_tcga]),
  ,
  drop = FALSE
]

# Gruppierung wie im urspruenglichen TCGA-Skript.
tcga$Group <- ifelse(
  !is.na(tcga$ERG.y) &
    grepl("TMPRSS2-ERG", tcga$ERG.y),
  "TMPRSS2-ERG",
  "no structural variant"
)

tcga$Group <- factor(
  tcga$Group,
  levels = group_levels
)

# ============================================================
# 4. Gruppengrößen ausgeben
# ============================================================

cat("\n==================================================\n")
cat("SUC Gruppengroessen\n")
cat("==================================================\n")
print(table(suc$Group))

cat("\n==================================================\n")
cat("TCGA Gruppengroessen\n")
cat("==================================================\n")
print(table(tcga$Group))

# ============================================================
# 5. Wilcoxon-Tests
# ============================================================

get_wilcox_results <- function(data, genes) {
  results <- lapply(
    genes,
    function(gene) {
      test_result <- wilcox.test(
        data[[gene]] ~ data$Group
      )

      data.frame(
        p_value = test_result$p.value,
        W_statistic = unname(test_result$statistic)
      )
    }
  )

  results <- do.call(rbind, results)

  results$Gene <- gene_names
  results$p_adj_BH <- p.adjust(
    results$p_value,
    method = "BH"
  )

  results[, c(
    "Gene",
    "p_value",
    "p_adj_BH",
    "W_statistic"
  )]
}

wilcox_suc <- get_wilcox_results(
  suc,
  genes_log2_suc
)

wilcox_tcga <- get_wilcox_results(
  tcga,
  genes_log2_tcga
)

cat("\n==================================================\n")
cat("Wilcoxon-Ergebnisse SUC\n")
cat("==================================================\n")
print(wilcox_suc)

cat("\n==================================================\n")
cat("Wilcoxon-Ergebnisse TCGA\n")
cat("==================================================\n")
print(wilcox_tcga)

# ============================================================
# 6. Datensatzspezifische Y-Achsen
# ============================================================

axis_settings <- list(
  "SUC" = list(
    limits = c(0, 15),
    breaks = seq(0, 15, by = 5),
    y_line = 13.75,
    y_bracket_bottom = 13.4,
    y_p = 14.5
  ),
  "TCGA" = list(
    limits = c(10, 25),
    breaks = seq(10, 25, by = 5),
    y_line = 23.75,
    y_bracket_bottom = 23.4,
    y_p = 24.5
  )
)

cat("\n==================================================\n")
cat("Datensatzspezifische Y-Achsen\n")
cat("==================================================\n")
cat("SUC: 0-15 | TCGA: 10-25 | Tick-Abstand: 5\n")

# ============================================================
# 7. p-Wert als Signifikanzsterne formatieren
# ============================================================

format_p_stars <- function(p_value) {
  if (is.na(p_value)) {
    return("ns")
  }

  if (p_value < 0.001) {
    return("***")
  }

  if (p_value < 0.01) {
    return("**")
  }

  if (p_value < 0.05) {
    return("*")
  }

  "ns"
}

# ============================================================
# 8. Einheitliche Plot-Funktion
# ============================================================

plot_gene <- function(
    data,
    gene_col,
    gene_title,
    dataset_name,
    show_y_title = TRUE
) {
  axis_config <- axis_settings[[dataset_name]]

  if (is.null(axis_config)) {
    stop("No axis settings defined for dataset: ", dataset_name)
  }

  test_result <- wilcox.test(
    data[[gene_col]] ~ data$Group
  )

  p_label <- format_p_stars(
    test_result$p.value
  )

  n_control <- sum(
    data$Group == "no structural variant"
  )

  n_fusion <- sum(
    data$Group == "TMPRSS2-ERG"
  )

  x_labels <- setNames(
    as.expression(list(
      bquote(atop("no structural variant", paste("n = ", .(n_control)))),
      bquote(atop(italic("TMPRSS2-ERG"), paste("n = ", .(n_fusion))))
    )),
    group_levels
  )

  y_line <- axis_config$y_line
  y_bracket_bottom <- axis_config$y_bracket_bottom
  y_p <- axis_config$y_p

  y_axis_title <- if (show_y_title) {
    "log2(mRNA expression + 1)"
  } else {
    NULL
  }

  p <- ggplot(
    data,
    aes(
      x = Group,
      y = .data[[gene_col]],
      fill = Group
    )
  ) +
    geom_violin(
      trim = FALSE,
      alpha = 0.35,
      color = NA
    ) +
    geom_boxplot(
      width = 0.18,
      outlier.shape = NA,
      color = "black",
      alpha = 0.85,
      coef = 0
    ) +
    geom_jitter(
      width = 0.20,
      size = 0.8,
      alpha = 0.65,
      color = "black"
    ) +
    stat_summary(
      fun = mean,
      geom = "point",
      shape = 21,
      size = summary_marker_size,
      stroke = summary_marker_stroke,
      position = position_nudge(x = -0.035),
      color = "black",
      fill = "white"
    ) +
    stat_summary(
      fun = median,
      geom = "point",
      shape = 21,
      size = summary_marker_size,
      stroke = summary_marker_stroke,
      position = position_nudge(x = 0.035),
      color = "black",
      fill = "#D62728"
    ) +
    scale_fill_manual(
      values = group_colors
    ) +
    scale_x_discrete(
      labels = x_labels
    ) +
    scale_y_continuous(
      breaks = axis_config$breaks,
      labels = as.character(axis_config$breaks),
      expand = c(0, 0)
    ) +
    coord_cartesian(
      ylim = axis_config$limits,
      clip = "off"
    ) +
    labs(
      title = NULL,
      x = NULL,
      y = y_axis_title
    ) +
    annotate(
      "segment",
      x = 1,
      xend = 2,
      y = y_line,
      yend = y_line,
      linewidth = 0.4
    ) +
    annotate(
      "segment",
      x = 1,
      xend = 1,
      y = y_line,
      yend = y_bracket_bottom,
      linewidth = 0.4
    ) +
    annotate(
      "segment",
      x = 2,
      xend = 2,
      y = y_line,
      yend = y_bracket_bottom,
      linewidth = 0.4
    ) +
    annotate(
      "text",
      x = 1.5,
      y = y_p,
      label = p_label,
      size = 13 / ggplot2::.pt
    ) +
    theme_classic(
      base_size = 13
    ) +
    theme(
      legend.position = "none",
      plot.title = element_text(
        face = "bold",
        hjust = 0.5
      ),
      plot.title.position = "panel",
      axis.title.y = element_text(
        face = "plain"
      ),
      axis.text = element_text(
        color = "black",
        size = 13
      ),
      axis.text.x = element_text(
        face = "plain",
        size = 13,
        lineheight = 1.0,
        margin = margin(
          t = 6
        )
      ),
      axis.line = element_line(
        linewidth = 0.4
      ),
      axis.ticks = element_line(
        linewidth = 0.4
      ),
      plot.margin = margin(
        t = 8,
        r = 8,
        b = 22,
        l = 8
      )
    )

  return(p)
}

# ============================================================
# 9. Alle acht Diagramme erzeugen
# ============================================================

suc_plots <- lapply(
  seq_along(genes_log2_suc),
  function(i) {
    plot_gene(
      data = suc,
      gene_col = genes_log2_suc[i],
      gene_title = gene_names[i],
      dataset_name = "SUC",
      show_y_title = FALSE
    )
  }
)

tcga_plots <- lapply(
  seq_along(genes_log2_tcga),
  function(i) {
    plot_gene(
      data = tcga,
      gene_col = genes_log2_tcga[i],
      gene_title = gene_names[i],
      dataset_name = "TCGA",
      show_y_title = TRUE
    )
  }
)

# ============================================================
# 10. TCGA links / SUC rechts
# ============================================================
#
# Die beiden Plot-GTables werden horizontal zu EINER GTable verbunden.
# Ihre Panelspalten liegen dadurch im selben Layout und besitzen beide
# die Breite 1null. So werden die X-Achsen garantiert gleich lang.
# ============================================================

combine_plot_pair <- function(tcga_plot, suc_plot) {
  tcga_grob <- ggplotGrob(tcga_plot)
  suc_grob <- ggplotGrob(suc_plot)

  shared_heights <- grid::unit.pmax(
    tcga_grob$heights,
    suc_grob$heights
  )

  panel_rows <- unique(
    tcga_grob$layout$t[tcga_grob$layout$name == "panel"]
  )
  shared_heights[panel_rows] <- panel_height

  suc_grob$heights <- shared_heights
  tcga_grob$heights <- shared_heights

  gtable:::cbind_gtable(
    tcga_grob,
    suc_grob,
    "first"
  )
}

paired_rows <- Map(
  combine_plot_pair,
  tcga_plots,
  suc_plots
)

combined_rows <- lapply(
  seq_along(gene_names),
  function(i) {
    paired_row <- paired_rows[[i]]

    # Die Genbezeichnung wird vertikal in der Panelzeile zentriert.
    panel_row <- paired_row$layout$t[
      paired_row$layout$name == "panel"
    ][1]

    paired_row <- gtable::gtable_add_cols(
      paired_row,
      gene_label_width,
      pos = 0
    )

    gene_label <- grid::textGrob(
      gene_names[i],
      gp = grid::gpar(
        fontface = "bold.italic",
        fontsize = 15
      ),
      just = c("center", "center")
    )

    gtable::gtable_add_grob(
      paired_row,
      gene_label,
      t = panel_row,
      b = panel_row,
      l = 1,
      r = 1,
      clip = "off"
    )
  }
)

# ============================================================
# 11. Überschriften
# ============================================================

# Die Überschriften werden ueber den tatsächlichen Panelspalten platziert.
panel_columns <- combined_rows[[1]]$layout$l[
  combined_rows[[1]]$layout$name == "panel"
]

tcga_panel_col <- panel_columns[1]
suc_panel_col <- panel_columns[2]

header_pair <- gtable::gtable(
  widths = combined_rows[[1]]$widths,
  heights = header_height
)

header_pair <- gtable::gtable_add_grob(
  header_pair,
  grid::textGrob(
    "TCGA",
    gp = grid::gpar(fontface = "bold", fontsize = 15),
    y = unit(2, "pt"),
    just = c("center", "bottom")
  ),
  t = 1,
  l = tcga_panel_col,
  clip = "off"
)

# Gemeinsame Legende für die Lageparameter, ohne die Panels zu verkleinern.
summary_key <- grid::grobTree(
  grid::pointsGrob(
    x = unit(1, "npc") - unit(59, "pt"),
    y = unit(0.72, "npc"),
    pch = 21,
    size = unit(7, "pt"),
    gp = grid::gpar(
      col = "black",
      fill = "white",
      lwd = summary_marker_stroke
    )
  ),
  grid::textGrob(
    "Mean",
    x = unit(1, "npc") - unit(49, "pt"),
    y = unit(0.72, "npc"),
    just = "left",
    gp = grid::gpar(fontsize = 13)
  ),
  grid::pointsGrob(
    x = unit(1, "npc") - unit(59, "pt"),
    y = unit(0.40, "npc"),
    pch = 21,
    size = unit(7, "pt"),
    gp = grid::gpar(
      col = "black",
      fill = "#D62728",
      lwd = summary_marker_stroke
    )
  ),
  grid::textGrob(
    "Median",
    x = unit(1, "npc") - unit(49, "pt"),
    y = unit(0.40, "npc"),
    just = "left",
    gp = grid::gpar(fontsize = 13)
  )
)

header_pair <- gtable::gtable_add_grob(
  header_pair,
  summary_key,
  t = 1,
  l = tcga_panel_col,
  r = suc_panel_col,
  clip = "off"
)

header_pair <- gtable::gtable_add_grob(
  header_pair,
  grid::textGrob(
    "SUC",
    gp = grid::gpar(fontface = "bold", fontsize = 15),
    y = unit(2, "pt"),
    just = c("center", "bottom")
  ),
  t = 1,
  l = suc_panel_col,
  clip = "off"
)

# ============================================================
# 12. Finale Abbildung
# ============================================================

final_plot <- arrangeGrob(
  grid::nullGrob(),
  header_pair,
  arrangeGrob(
    grobs = combined_rows,
    ncol = 1
  ),
  ncol = 1,
  heights = unit.c(
    header_top_padding,
    header_height,
    unit(1, "null")
  )
)

# ============================================================
# 13. Abbildung anzeigen
# ============================================================

grid.newpage()

grid.draw(
  final_plot
)

# ============================================================
# 14. Abbildung als PDF speichern
# ============================================================

ggsave(
  filename = "Figure_Violin_SUC_TCGA_FINAL_compact_axes_new.pdf",
  plot = final_plot,
  width = 10,
  height = 12,
  units = "in",
  dpi = 600
)
