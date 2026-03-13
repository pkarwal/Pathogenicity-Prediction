import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("new_data.csv")

df["label"] = df["Germline classification"].map({
    "Benign": 0,
    "Pathogenic": 1
})

df["aa_substitution"] = df["aa_from"] + "→" + df["aa_to"]
aa_stats = (
    df.groupby("aa_substitution")
      .agg(
          total_cases=("label", "count"),
          pathogenic_cases=("label", "sum")
      )
)

aa_stats["pathogenic_rate"] = aa_stats["pathogenic_cases"] / aa_stats["total_cases"]

# Only keep substitutions with enough samples
aa_stats = aa_stats[aa_stats["total_cases"] >= 3]

aa_stats = aa_stats.sort_values(
    ["pathogenic_rate", "total_cases"],
    ascending=[False, False]
)

pos_stats = (
    df.groupby("aa_pos")
      .agg(
          total_cases=("label", "count"),
          pathogenic_cases=("label", "sum")
      )
)

pos_stats["pathogenic_rate"] = pos_stats["pathogenic_cases"] / pos_stats["total_cases"]

pos_stats = pos_stats[pos_stats["total_cases"] >= 3]

pos_stats = pos_stats.sort_values(
    ["pathogenic_rate", "total_cases"],
    ascending=[False, False]
)

full_pos_stats = (
    df.groupby("aa_pos")
      .agg(
          total_cases=("label", "count"),
          pathogenic_cases=("label", "sum")
      )
)

plt.figure(figsize=(10,6))

# plot all positions in one color
plt.scatter(
    full_pos_stats.index,
    full_pos_stats["pathogenic_cases"],
    s=full_pos_stats["pathogenic_cases"] * 40,
    alpha=0.6,
    color="steelblue",
    edgecolor="black"
)

# highlight position 835
hotspot_835 = full_pos_stats.loc[835]

plt.scatter(
    835,
    hotspot_835["pathogenic_cases"],
    s=hotspot_835["pathogenic_cases"] * 60,
    color="red",
    edgecolor="black",
    label="Position 835 hotspot"
)

plt.xlabel("Amino Acid Position")
plt.ylabel("Number of Pathogenic Mutations")
plt.title("FLT3 Pathogenic Mutation Hotspots")
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("hotspots.png", dpi=300, bbox_inches="tight")


df["protein_mutation"] = (
    df["aa_from"] + df["aa_pos"].astype(str) + df["aa_to"]
) 
mutation_stats = (
    df.groupby("protein_mutation")
      .agg(
          total_cases=("label", "count"),
          pathogenic_cases=("label", "sum")
      )
)

mutation_stats["pathogenic_rate"] = (
    mutation_stats["pathogenic_cases"] / mutation_stats["total_cases"]
)

mutation_stats = mutation_stats[mutation_stats["total_cases"] >= 2]

mutation_stats = mutation_stats.sort_values(
    ["pathogenic_rate", "total_cases"],
    ascending=[False, False]
)

print(mutation_stats.head(15))