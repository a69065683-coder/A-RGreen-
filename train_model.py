"""تدريب نموذج تصنيف أمراض أوراق النباتات من PlantVillage عبر TFDS.

التشغيل:
    python train_model.py --epochs 12 --batch-size 32
الناتج: model.h5 و model_labels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = (224, 224)
# أسماء الفئات كما تظهر في PlantVillage/TFDS.
TARGETS = [
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Squash___Powdery_mildew",
]
OUTPUT_LABELS = ["tomato_late_blight", "potato_early_blight", "potato_late_blight", "powdery_mildew"]


def build_dataset(batch_size: int, seed: int):
    ds, info = tfds.load("plant_village", split="train", as_supervised=True, with_info=True, shuffle_files=True)
    names = info.features["label"].names
    missing = [x for x in TARGETS if x not in names]
    if missing:
        raise RuntimeError(f"لم توجد الفئات التالية في نسخة PlantVillage: {missing}. الفئات المتاحة: {names}")
    wanted_ids = [names.index(x) for x in TARGETS]
    id_to_output = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(wanted_ids, dtype=tf.int64), tf.range(len(wanted_ids), dtype=tf.int64)),
        default_value=-1,
    )

    def keep(image, label):
        return tf.reduce_any(tf.equal(tf.cast(label, tf.int64), tf.constant(wanted_ids, dtype=tf.int64)))

    def prepare(image, label):
        image = tf.image.resize(image, IMG_SIZE) / 255.0
        return image, tf.cast(id_to_output.lookup(tf.cast(label, tf.int64)), tf.int32)

    ds = ds.filter(keep).map(prepare, num_parallel_calls=tf.data.AUTOTUNE).shuffle(12000, seed=seed, reshuffle_each_iteration=False)
    count = int(tf.data.experimental.cardinality(ds).numpy())
    if count < 20:
        raise RuntimeError("عدد الصور المختارة قليل جدًا للتدريب.")
    val_count = max(1, int(count * 0.2))
    val_ds = ds.take(val_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = ds.skip(val_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, count


def make_model(num_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, count = build_dataset(args.batch_size, args.seed)
    model = make_model(len(TARGETS))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(out / "model.h5", monitor="val_accuracy", save_best_only=True),
    ]
    print(f"Training on {count} selected images across {len(TARGETS)} classes")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(out / "model.h5")
    (out / "model_labels.json").write_text(json.dumps(OUTPUT_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out / 'model.h5'} and {out / 'model_labels.json'}")


if __name__ == "__main__":
    main()
