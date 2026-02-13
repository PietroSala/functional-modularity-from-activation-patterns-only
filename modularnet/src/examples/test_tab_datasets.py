
# =============================================================================
# Example Usage
# =============================================================================

from dataprovider.dataprovider import CovertypeDataProvider, HIGGSDataProvider


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Covertype DataProvider")
    print("=" * 60)
    
    # Create Covertype provider
    covertype = CovertypeDataProvider(
        batch_size=128,
        split=[0.7, 0.15, 0.15],
        normalize=True,
        num_workers=2
    )
    
    print(f"Dataset: {covertype.name}")
    print(f"Num features: {covertype.num_features}")
    print(f"Num classes: {covertype.num_classes}")
    print(f"Train batches: {len(covertype.train)}")
    print(f"Val batches: {len(covertype.val)}")
    print(f"Test batches: {len(covertype.test)}")
    
    # Get a batch
    x, y = next(iter(covertype.train))
    print(f"\nBatch shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Label range: {y.min()}-{y.max()}")
    print(f"Feature groups: {covertype.get_feature_groups()}")
    
    print("\n" + "=" * 60)
    print("Testing HIGGS DataProvider (with subset)")
    print("=" * 60)
    
    # Create HIGGS provider with subset
    higgs = HIGGSDataProvider(
        batch_size=128,
        split=[0.7, 0.15, 0.15],
        normalize=True,
        subset_size=100000,  # Use 100k samples for quick testing
        num_workers=2
    )
    
    print(f"Dataset: {higgs.name}")
    print(f"Num features: {higgs.num_features}")
    print(f"Num classes: {higgs.num_classes}")
    print(f"Train batches: {len(higgs.train)}")
    print(f"Val batches: {len(higgs.val)}")
    print(f"Test batches: {len(higgs.test)}")
    
    # Get a batch
    x, y = next(iter(higgs.train))
    print(f"\nBatch shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Label range: {y.min()}-{y.max()}")
    print(f"Feature groups: {higgs.get_feature_groups()}")