import unittest
from dataset_audit import DatasetAuditor

class TestDatasetSplitting(unittest.TestCase):
    def test_group_aware_split_prevents_leakage(self):
        auditor = DatasetAuditor()
        
        # Mock records: 3 noisy variants of the same clean image (hash1)
        # and 2 noisy variants of another clean image (hash2)
        auditor.records = [
            {"image_id": "img1_n1", "dataset_name": "DS1", "clean_hash": "hash1", "noisy_hash": "nh1", "split": ""},
            {"image_id": "img1_n2", "dataset_name": "DS1", "clean_hash": "hash1", "noisy_hash": "nh2", "split": ""},
            {"image_id": "img1_n3", "dataset_name": "DS1", "clean_hash": "hash1", "noisy_hash": "nh3", "split": ""},
            {"image_id": "img2_n1", "dataset_name": "DS2", "clean_hash": "hash2", "noisy_hash": "nh4", "split": ""},
            {"image_id": "img2_n2", "dataset_name": "DS2", "clean_hash": "hash2", "noisy_hash": "nh5", "split": ""}
        ]
        
        # Execute the group-aware split algorithm
        auditor.group_aware_split()
        
        # Verify that all variants of hash1 share the EXACT same split
        splits_hash1 = set([r["split"] for r in auditor.records if r["clean_hash"] == "hash1"])
        self.assertEqual(len(splits_hash1), 1, "Leakage detected! Variants of hash1 crossed splits.")
        
        # Verify that all variants of hash2 share the EXACT same split
        splits_hash2 = set([r["split"] for r in auditor.records if r["clean_hash"] == "hash2"])
        self.assertEqual(len(splits_hash2), 1, "Leakage detected! Variants of hash2 crossed splits.")
        
        # Run the internal leakage detector to ensure it registers 0 critical errors
        auditor.detect_leakage()
        self.assertEqual(auditor.stats["cross_split_leakage"], 0)
        self.assertEqual(auditor.stats["leakage_train_val"], 0)
        self.assertEqual(auditor.stats["leakage_train_test"], 0)
        self.assertEqual(auditor.stats["leakage_val_test"], 0)

if __name__ == "__main__":
    unittest.main()
