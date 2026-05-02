from rag.pipeline import RAGPipeline

rag = RAGPipeline()

rag.index_text("dataset", limit=30)
rag.index_images(
    "dataset/Plant_leaf_diseases_dataset_without_augmentation",
    limit=20
)

print("🎉 ALL INDEXING DONE")
