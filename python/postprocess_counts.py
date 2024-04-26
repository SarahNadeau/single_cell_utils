import argparse
import logging

import pandas as pd
import numpy as np
import pyranges as pr
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def parse_args():
    """Set up the parsing of command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Postprocess counts in adata format by filtering cells and genes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-outfile",
        required=False,
        default="postprocessed.adata",
        type=str,
        dest="outfile",
        help="Path to output file",
    )
    parser.add_argument(
        "-log",
        required=False,
        default="postprocessing.log",
        type=str,
        dest="log",
        help="Path to log file",
    )
    parser.add_argument(
        "-adata",
        required=False,
        dest="adata",
        type=str,
        help="Path to adata count file",
    )
    parser.add_argument(
        "-metadata",
        required=False,
        dest="metadata",
        type=str,
        help="Path to metadata file",
    )
    parser.add_argument(
        "-metadata-sep",
        default=",",
        required=False,
        dest="metadata_sep",
        type=str,
        help="Separator for metadata file",
    )
    parser.add_argument(
        "-gtf",
        required=False,
        dest="gtf",
        type=str,
        help="Path to GTF genome annotations file",
    )
    parser.add_argument(
        "-pct_mt_threshold",
        default=10,
        dest="pct_mt_threshold",
        type=float,
        help="Upper bound for percent total counts mitochondrial",
    )
    parser.add_argument(
        "-pct_in_genes_threshold",
        default=50,
        dest="pct_in_genes_threshold",
        type=float,
        help="Lower bound for percent mapped mapped reads in genes",
    )
    parser.add_argument(
        "-cells_to_remove",
        required=False,
        dest="cells_to_remove",
        nargs="+",
        type=str,
        help="Comma-separated list of cells to remove",
    )
    parser.add_argument(
        "-genes_to_remove",
        required=False,
        dest="genes_to_remove",
        nargs="+",
        type=str,
        help="Comma-separated list of genes to remove",
    )
    parser.add_argument(
        "-genes_to_normalize_with",
        required=False,
        dest="genes_to_normalize_with",
        nargs="+",
        type=str,
        help="Comma-separated list of genes to normalize with, default is all remaining genes",
    )
    parser.add_argument(
        "-norm_sum",
        default=100000,
        dest="norm_sum",
        type=int,
        help="Total counts to normalize to",
    )
    return parser.parse_args()


def concat_adatas(adata1, adata2, outfile=None, label=None, keys=None):
    adata1 = sc.read_h5ad(adata1)
    logger.info(f"Loaded adata1 with {adata1.n_obs} obs and {adata1.n_vars} vars")
    adata2 = sc.read_h5ad(adata2)
    logger.info(f"Loaded adata2 with {adata2.n_obs} obs and {adata2.n_vars} vars")

    adata = ad.concat([adata1, adata2], join="outer", label=label, keys=keys)
    if outfile is not None:
        adata.write_h5ad(outfile)
    else:
        return adata


def load_data(adata, metadata=None, metadata_sep=",", metadata_colname_sample="sample", gtf=None):
    logger.info("Loading adata from file")
    adata = sc.read_h5ad(adata)

    if metadata:
        logger.info("Loading and merging metadata")
        metadata = pd.read_csv(metadata, sep=metadata_sep, dtype={metadata_colname_sample: str})

        # Add metadata to count data
        adata.obs["cell_id"] = adata.obs.index
        adata.obs = adata.obs.merge(
            right=metadata, how="left", left_on="cell_id", right_on=metadata_colname_sample
        )

    if gtf:
        logger.info("Loading and merging genome annotations")
        annotations = pr.read_gtf(gtf)

        # Add gene names to count data
        gene_name_mapping = annotations[annotations.Feature == "gene"].df[
            ["gene_id", "gene_name"]
        ]
        adata.var = adata.var.merge(
            right=gene_name_mapping, how="left", left_on="Geneid", right_on="gene_id"
        )
        # Use gene_id if there's no gene_name in annotations
        adata.var.gene_name.fillna("", inplace=True)
        j = 0
        for i in range(adata.var.shape[0]):
            gene_name = adata.var.iloc[i]["gene_name"]
            if gene_name == "":
                j += 1
                gene_id = adata.var.iloc[i]["gene_id"]
                adata.var.iat[i, 6] = gene_id

        # Make variables index gene names
        adata.var.set_index("gene_name", inplace=True, drop=False)
        adata.var_names_make_unique()

        logger.info(
            f"{j} out of {i + 1} genes had no gene_name entry, using gene_ids instead"
        )

    return adata


def filter_cells(
    adata, pct_mt_threshold=10, pct_in_genes_threshold=50, cells_to_remove=None
):
    # Filter out cells with >X% mitochondrial genes
    n_total_cells = adata.shape[0]
    n_filtered_cells = sum(adata.obs.pct_counts_mt > pct_mt_threshold)
    adata = adata[adata.obs.pct_counts_mt <= pct_mt_threshold, :]
    logger.info(
        f"{n_filtered_cells} out of {n_total_cells} cells discarded for having > {pct_mt_threshold}% mt genes"
    )

    # Filter out cells with >X% reads mapping intergenic regions
    if ("n_reads_mapped_genes" in adata.obs.columns) & ("n_total_mappings" in adata.obs.columns):
        n_total_cells = adata.shape[0]
        n_filtered_cells = sum(
            np.divide(adata.obs.n_reads_mapped_genes, adata.obs.n_total_mappings)
            < pct_in_genes_threshold / 100
        )
        adata.obs["to_remove"] = list(
            np.divide(adata.obs.n_reads_mapped_genes, adata.obs.n_total_mappings)
            < pct_in_genes_threshold / 100
        )
        adata = adata[adata.obs.to_remove == False, :]
        logger.info(
            f"{n_filtered_cells} out of {n_total_cells} cells discarded for having < {pct_in_genes_threshold}% reads mapping in genes"
        )
    else:
        logger.info(
            f"Not filtering based on percent reads mapping in genes because one of adata obs columns 'n_reads_mapped_genes' and 'n_total_mappings' is missing"
        )

    # Filter out cells in to-remove list
    if cells_to_remove:
        adata.obs["to_remove"] = adata.obs.cell_id.isin(cells_to_remove).values
        adata = adata[adata.obs.to_remove == False, :]
        logger.info(
            f"{sum(adata.obs.to_remove)} cells discarded for being in to-remove list"
        )

    adata.raw = adata
    return adata


def filter_genes(adata, gene_colname="gene_name", genes_to_remove=None, genes_to_normalize_with=None, min_cells=None):
    if gene_colname in adata.var.columns:
        logger.info(
            f"Using gene names in adata.var column {gene_colname}"
        )
    else:
        logger.info(
            f"Adding column {gene_colname} to adata.var with index values to facilitate gene filtering"
        )
        adata.var[gene_colname] = adata.var.index.values

    # Filter out mitochondrial and ribosomal genes: these are confounders because they are correlated with bad quality (both are large and leak slower through membrane)
    n_removed_rps = sum(adata.var[gene_colname].str.startswith("Rps"))
    n_removed_rpl = sum(adata.var[gene_colname].str.startswith("Rpl"))
    n_removed_mt = sum(adata.var[gene_colname].str.startswith("mt-"))
    adata = adata[:, ~adata.var[gene_colname].str.startswith("Rps")]
    adata = adata[:, ~adata.var[gene_colname].str.startswith("Rpl")]
    adata = adata[:, ~adata.var[gene_colname].str.startswith("mt-")]
    logger.info(
        f"{n_removed_rpl + n_removed_rps + n_removed_mt} genes discarded for being auto-detected mitochondrial or ribosomal"
    )

    # Filter out genes we don't want to affect normalization
    if not genes_to_remove is None:
        adata.var["to_remove"] = adata.var[gene_colname].isin(genes_to_remove).values
        n_to_remove = sum(adata.var.to_remove)
        adata = adata[:, ~adata.var.to_remove]
        logger.info(f"{n_to_remove} genes to remove removed")

    # Optionally filter to only certain genes for normalization
    if not genes_to_normalize_with is None:
        adata.var["gene_name_upper"] = adata.var[gene_colname].str.upper()
        adata.var["to_normalize_with"] = adata.var["gene_name_upper"].isin(genes_to_normalize_with)

        n_to_normalize_with = sum(adata.var["to_normalize_with"])
        adata = adata[:, adata.var.to_normalize_with]

        logger.info(
            f"Filtered to {n_to_normalize_with} genes specificd to normalize with"
        )
    
    # Optionally filter out genes present in fewer than min_cells cells
    if not min_cells is None:
        n_genes_before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells)
        n_genes_filtered = n_genes_before - adata.n_vars
        logger.info(
            f"Filtered out {n_genes_filtered} genes present in < {min_cells} cells"
        )

    logger.info(
        f"{adata.n_vars} genes remain after gene-based filtering"
    )

    return adata


def normalize_counts(adata, norm_sum=100000):
    # Total count normalize reads per cell, so that counts become comparable among cells
    # But only include non-filtered genes in library size calculation
    logger.info(f"Normalizing counts using gene set with {adata.n_vars} genes")

    adata_norm = adata.copy()
    library_size = np.sum(adata_norm.X, axis=1)
    adata_norm.X = np.divide(adata_norm.X, library_size[:, None]) * norm_sum

    adata_norm.obs["library_size_used_to_normalize"] = library_size
    adata_norm.var["normalization_sum"] = norm_sum

    return adata_norm


def main():
    args = parse_args()
    logging.basicConfig(filename=args.log, level=logging.INFO)

    adata = load_data(
        adata=args.adata,
        metadata=args.metadata,
        metadata_sep=args.metadata_sep,
        gtf=args.gtf,
    )

    adata = filter_cells(adata=adata, cells_to_remove=args.cells_to_remove)

    adata = filter_genes(
        adata=adata,
        genes_to_remove=args.genes_to_remove,
        genes_to_normalize_with=args.genes_to_normalize_with,
    )

    adata = normalize_counts(adata=adata, norm_sum=args.norm_sum)

    adata.write_h5ad(args.outfile)


if __name__ == "__main__":
    main()
