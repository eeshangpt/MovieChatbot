"""Integration test for run_training_pipeline using a monkeypatched embedder
(no network access needed for Tier1a). Tier2 cascade eval is expected to be
skipped gracefully since the NLI model can't be downloaded in this sandbox.
"""

import pathlib
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import relevance_gate.embedding_classifier as embedding_classifier_module
from relevance_gate.data_io import save_training_examples
from relevance_gate.schemas import Label, LabelSource, TrainingExample
from relevance_gate.train_pipeline import run_training_pipeline


def make_fake_embedder(dim=16):
    class FakeEmbedder:
        def encode(
            self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        ):
            out = []
            for t in texts:
                local_rng = np.random.RandomState(abs(hash(t)) % (2**32))
                vec = local_rng.rand(dim)
                if "movie" in t.lower() or "film" in t.lower():
                    vec[:4] += 5.0
                out.append(vec)
            return np.array(out)

    return FakeEmbedder()


def main():
    # Monkeypatch at the class level so instances created *inside*
    # run_training_pipeline also get the fake embedder.
    original_load_embedder = (
        embedding_classifier_module.EmbeddingLRClassifier._load_embedder
    )
    embedding_classifier_module.EmbeddingLRClassifier._load_embedder = lambda self: (
        make_fake_embedder()
    )

    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    try:
        examples = [
            TrainingExample(
                "what movie won best picture in 2020",
                Label.RELEVANT,
                LabelSource.SYNTHETIC,
            ),
            TrainingExample(
                "recommend a good film tonight", Label.RELEVANT, LabelSource.SYNTHETIC
            ),
            TrainingExample(
                "best film about space travel", Label.RELEVANT, LabelSource.SYNTHETIC
            ),
            TrainingExample(
                "who directed that movie about robots",
                Label.RELEVANT,
                LabelSource.SYNTHETIC,
            ),
            TrainingExample(
                "is that film worth watching", Label.RELEVANT, LabelSource.SYNTHETIC
            ),
            TrainingExample(
                "when did that movie release", Label.RELEVANT, LabelSource.SYNTHETIC
            ),
            TrainingExample(
                "what's the weather in paris",
                Label.OUT_OF_DOMAIN,
                LabelSource.SYNTHETIC,
            ),
            TrainingExample(
                "how do i boil an egg", Label.OUT_OF_DOMAIN, LabelSource.SYNTHETIC
            ),
            TrainingExample(
                "is it going to rain tomorrow",
                Label.OUT_OF_DOMAIN,
                LabelSource.SYNTHETIC,
            ),
            TrainingExample(
                "help me with my math homework",
                Label.OUT_OF_DOMAIN,
                LabelSource.SYNTHETIC,
            ),
            TrainingExample(
                "what's the best stock to buy",
                Label.OUT_OF_DOMAIN,
                LabelSource.SYNTHETIC,
            ),
            TrainingExample(
                "how do i fix a flat tire", Label.OUT_OF_DOMAIN, LabelSource.SYNTHETIC
            ),
        ]
        data_path = tmp_dir / "synthetic.jsonl"
        save_training_examples(examples, data_path)
        print(f"OK: saved {len(examples)} examples to {data_path}")

        output_dir = tmp_dir / "artifacts"
        mlflow_uri = f"sqlite:///{tmp_dir / 'mlflow.db'}"

        summary = run_training_pipeline(
            output_dir=output_dir,
            synthetic_data_path=data_path,
            production_log_path=None,
            test_split=0.34,
            train_finetuned=False,
            mlflow_tracking_uri=mlflow_uri,
            mlflow_experiment_name="smoke-test-experiment",
            evaluate_full_cascade=True,  # expected to gracefully skip (no Tier2 model access)
            random_state=0,
        )

        assert "run_id" in summary
        print(f"OK: MLflow run completed, run_id={summary['run_id']}")

        assert "tier1a_metrics" in summary
        m = summary["tier1a_metrics"]
        assert 0.0 <= m["accuracy"] <= 1.0
        print(f"OK: tier1a metrics present, accuracy={m['accuracy']:.3f}")

        if "cascade_metrics" in summary:
            print(
                "NOTE: cascade eval unexpectedly succeeded (Tier2 model was reachable)."
            )
        else:
            print(
                "OK: cascade eval gracefully skipped (Tier2 model unavailable, as expected)."
            )

        lr_model_dir = output_dir / "tier1_lr"
        assert (lr_model_dir / "lr_head.joblib").exists()
        assert (lr_model_dir / "metadata.json").exists()
        print("OK: Tier1a model artifacts saved to disk.")

        import mlflow as mlflow_module

        mlflow_module.set_tracking_uri(mlflow_uri)
        run = mlflow_module.get_run(summary["run_id"])
        assert run.data.params.get("n_train") is not None
        assert run.data.metrics.get("tier1a_accuracy") is not None
        print("OK: MLflow run has logged params and metrics.")

        print("\nAll training pipeline integration tests passed.")
    finally:
        embedding_classifier_module.EmbeddingLRClassifier._load_embedder = (
            original_load_embedder
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
