
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf

df = pd.read_csv("processed_news.csv")
train, test = train_test_split(df, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def convert_data(df):
    return [InputExample(guid=None, text_a=text, text_b=None, label=label) for text, label in zip(df['clean_text'], df['label'])]

train_examples = convert_data(train)
test_examples = convert_data(test)

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []
    for e in examples:
        inputs = tokenizer.encode_plus(e.text_a,
                                       max_length=max_length,
                                       truncation=True,
                                       padding='max_length',
                                       add_special_tokens=True,
                                       return_token_type_ids=False)
        features.append(InputFeatures(input_ids=inputs['input_ids'],
                                      attention_mask=inputs['attention_mask'],
                                      label=e.label))
    def gen():
        for f in features:
            yield ({'input_ids': f.input_ids, 'attention_mask': f.attention_mask}, f.label)

    return tf.data.Dataset.from_generator(gen,
        ({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.int64),
        ({'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None])}, tf.TensorShape([])))

train_dataset = convert_examples_to_tf_dataset(train_examples, tokenizer)
train_dataset = train_dataset.shuffle(100).batch(32)

test_dataset = convert_examples_to_tf_dataset(test_examples, tokenizer)
test_dataset = test_dataset.batch(32)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, epochs=2, validation_data=test_dataset)
model.save_pretrained("bert_fakenews_model")
tokenizer.save_pretrained("bert_fakenews_model")
