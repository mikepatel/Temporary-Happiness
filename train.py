"""
"""
################################################################################
# Imports
from parameters import *
from model import build_model


################################################################################
# Main
if __name__ == "__main__":
    # print out TF version
    print(f'TF version: {tf.__version__}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    text = []
    labels = []

    # read in csv
    df = pd.read_csv(CSV_FILEPATH)

    # Clean dataset
    # SKIP FOR NOW - will change from using Twitter dataset in the future

    # get text sequences and category labels
    text = list(df["text"])
    labels = list(df["class"])

    # will make labels binary for now
    # 0 = not offenseive
    # 1 = offensive
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = 1
        if labels[i] == 2:
            labels[i] = 0

    # map class categories to integers
    # arbitrary class names
    class_names = [
        "not offensive",
        "offensive"
    ]

    num_classes = len(class_names)

    class2int = {class_names[i]: i for i in range(len(class_names))}

    # map integers to class categories
    int2class = {v: k for k, v in class2int.items()}

    # split data set into training, validation, test
    # SKIP FOR NOW
    # training
    train_text = text
    train_labels = np.array(labels)

    # validation

    # test

    #print(f'Number of training text examples: {len(train_text)}')
    #print(f'Number of training labels: {len(train_labels)}')

    # Tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_WORDS
    )

    # tokenize on training set
    tokenizer.fit_on_texts(train_text)

    word2int = tokenizer.word_index  # unique tokens
    vocab_size = len(word2int)

    # Vectorization
    train_text = tokenizer.texts_to_matrix(train_text)
    train_text = np.array(train_text)

    # one-hot encode labels
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)

    # print shape
    print(f'Shape of train text: {train_text.shape}')
    print(f'Shape of train labels: {train_labels.shape}')

    # ----- MODEL ----- #
    # build model
    model = build_model(num_categories=num_classes)
    model.summary()

    # loss function, optimizer
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # ----- TRAIN ----- #
    # train model
    history = model.fit(
        x=train_text,
        y=train_labels,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )

    # save model
    # SKIP FOR NOW

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(PLOT_FILEPATH)

    # ----- PREDICT ----- #
    test_text = [
        "Miami vs Lebron, here we go",
        "What the fuck is this shit?"
    ]

    # tokenize
    test_tokens = tokenizer.texts_to_matrix(test_text)
    test_tokens = np.array(test_tokens)

    # make predictions
    predictions = model.predict(test_tokens)

    for i in range(len(test_text)):
        print()
        print(f'Text: {test_text[i]}')
        print(f'Prediction: {int2class[np.argmax(predictions[i])]}')
