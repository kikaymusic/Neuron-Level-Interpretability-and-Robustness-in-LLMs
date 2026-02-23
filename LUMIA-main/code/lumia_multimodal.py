import numpy as np
from sklearn.model_selection import train_test_split
from lumia_multimodal_utils import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse
import os
import torch



if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Mia with hidden states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, help="dataset name")
    parser.add_argument("-m", "--model", type=str, help="model size")
    parser.add_argument("-g", "--gpu", type=str, help="gpu")
    parser.add_argument("-r", "--round", type=str, help="round")
    parser.add_argument("-s", "--samples", type=str, help="round")
    args = parser.parse_args()
    config = vars(args)
    epochs = 1000
    batch_size = 64
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(torch.cuda.is_available())
    device = torch.device("cuda")
    samples_analyze = int(args.samples)
    model, processor = load_multimodal(args.model)

    if 'textcap' in args.dataset:
        member_samples, not_member_sample = process_members_not_members_textcaps(int(args.round))
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_textcap(member_samples,not_member_sample, samples_analyze,model,processor)

    elif 'math' in args.dataset: 
        member_samples, not_member_sample = process_members_not_members(int(args.round))
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_math(member_samples,not_member_sample, samples_analyze,model,processor)

    elif 'science' in args.dataset:
        member_samples, not_member_sample = process_members_not_members_science(int(args.round))
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_science(member_samples,not_member_sample, samples_analyze,model,processor)

    elif 'ao2' in args.dataset:
        member_samples, not_member_sample = process_members_not_members_aok2(int(args.round))
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_aok(member_samples,not_member_sample, samples_analyze,model,processor)

    elif 'chartqa' in args.dataset:
        member_samples, not_member_sample = process_members_not_members_chartqa(int(args.round))
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_chart(member_samples,not_member_sample, samples_analyze,model,processor)
    
    elif 'magpie' in args.dataset:
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_magpie(samples_analyze,model,processor)
    
    elif 'iconqa' in args.dataset:
        member_samples, not_member_sample = process_members_not_members_icon(int(args.round))
        members_visual, members_lang, nonmembers_visual, nonmembers_lang = extract_samples_icon(member_samples,not_member_sample, samples_analyze,model,processor)        
    else:
        print("Non available dataset. Quitting")
        exit()

      
    print(f'Members language shape: {members_lang.shape}')
    print(f'Members visual shape: {members_visual.shape}')
    print(f'non members language shape: {nonmembers_lang.shape}')
    print(f'non Members visual shape: {nonmembers_visual.shape}')
    visual = 0
    if visual:
        activations_combined = np.concatenate(
                    (members_visual, nonmembers_visual)
                )
    else:
        activations_combined = np.concatenate(
                    (members_lang, nonmembers_lang)
                )       

    labels_member = [1] * len(members_lang)
    labels_nonmember = [0] * len(nonmembers_lang)
    labels_combined = labels_member + labels_nonmember
    labels_combined = np.array(labels_combined)



    X_train, X_test, y_train, y_test = train_test_split(
        activations_combined,
        labels_combined,
        test_size=0.2,
        stratify=labels_combined,
    
    )

    max_acc = 0
    max_auc = 0
    acc_arr_lang = []
    acc_auc_lang = []
    layer_arr_lang = []
    print('Language + visual')
    with tf.device('/gpu:0'):
        for layer in range(X_train.shape[1]):
            print(f"Extracting for layer {layer}")
            X_train_layer = []
            X_test_layer = []

            for sample in X_train:
                X_train_layer.append(np.array(sample[layer][0]))
            for sample in X_test:
                X_test_layer.append(np.array(sample[layer][0]))

            num_labels = 2
            X_train_layer = np.array(X_train_layer)
            X_test_layer = np.array(X_test_layer)


            y_train = np.array(y_train)
            y_test = np.array(y_test)

            classifier = build_classifier(X_train.shape[2])

            early_stopping = EarlyStopping(
                monitor="val_accuracy", 
                patience=100,  
                restore_best_weights=True, 
            )

            history = classifier.fit(
                        x=X_train_layer,
                        y=y_train,
                        validation_data=(X_test_layer, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=0,  # To suppress detailed output
                    )
            val_accuracies = history.history["val_accuracy"]

            best_val_accuracy = max(val_accuracies)
            max_acc = max(best_val_accuracy, max_acc)
            acc_arr_lang.append(best_val_accuracy)

            layer_arr_lang.append(layer)


            y_pred_probs = classifier.predict(X_test_layer)

            if (
                len(y_test.shape) == 1 or y_test.shape[1] == 1
            ):  
                if y_pred_probs.shape[1] == 2:
                    y_pred_probs = y_pred_probs[:, 1]

                auc_roc = roc_auc_score(y_test, y_pred_probs)
                print(f"AUC-ROC Score: {auc_roc:.4f}")
                max_auc = max(max_auc, auc_roc)
                print(f"MAX AUC-ROC Score: {max_auc:.4f}")
                acc_auc_lang.append(auc_roc)
            else:
                print("Multi-class AUC-ROC calculation is not shown here.")

        print("-----------------")

        print(f"Max acc is: {max_acc}")
        print(f"Max auc is: {max_auc}")
        print('Visual')

        visual = 1
        if visual:
            activations_combined = np.concatenate(
                        (members_visual, nonmembers_visual)
                    )
        else:
            activations_combined = np.concatenate(
                        (members_lang, nonmembers_lang)
                    )       

        labels_member = [1] * len(members_lang)
        labels_nonmember = [0] * len(nonmembers_lang)
        labels_combined = labels_member + labels_nonmember
        labels_combined = np.array(labels_combined)



        X_train, X_test, y_train, y_test = train_test_split(
            activations_combined,
            labels_combined,
            test_size=0.2,
            stratify=labels_combined,
        
        )
        print(X_train.shape)


        max_acc = 0
        max_auc = 0
        acc_arr = []
        acc_auc = []
        layer_arr_2 = []

        for layer in range(X_train.shape[1]):
            print(f"Extracting for layer {layer}")
            X_train_layer = []
            X_test_layer = []

            for sample in X_train:
                X_train_layer.append(np.array(sample[layer][0]))
            for sample in X_test:
                X_test_layer.append(np.array(sample[layer][0]))

            num_labels = 2
            X_train_layer = np.array(X_train_layer)
            X_test_layer = np.array(X_test_layer)

            y_train = np.array(y_train)
            y_test = np.array(y_test)

            classifier = build_classifier(X_train.shape[2])

            early_stopping = EarlyStopping(
                monitor="val_accuracy",  #
                patience=100,
                restore_best_weights=True,  
            )
            epochs = 3000
            batch_size = 64
            history = classifier.fit(
                        x=X_train_layer,
                        y=y_train,
                        validation_data=(X_test_layer, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=0, 
                        
                    )
            val_accuracies = history.history["val_accuracy"]

            best_val_accuracy = max(val_accuracies)

            max_acc = max(best_val_accuracy, max_acc)
            acc_arr.append(best_val_accuracy)

            layer_arr_2.append(layer)

            y_pred_probs = classifier.predict(X_test_layer)


            if (
                len(y_test.shape) == 1 or y_test.shape[1] == 1
            ):  
                if y_pred_probs.shape[1] == 2:
                    y_pred_probs = y_pred_probs[:, 1]

                # Calculate AUC-ROC
                auc_roc = roc_auc_score(y_test, y_pred_probs)
                print(f"AUC-ROC Score: {auc_roc:.4f}")
                max_auc = max(max_auc, auc_roc)
                print(f"MAX AUC-ROC Score: {max_auc:.4f}")
                acc_auc.append(auc_roc)
            else:
                # For multi-class classification, you may want to compute AUC for each class
                print("Multi-class AUC-ROC calculation is not shown here.")

    print("-----------------")

    print(f"Max acc is: {max_acc}")
    print(f"Max auc is: {max_auc}")
    df = pd.DataFrame()
    if len(layer_arr_2) > len(layer_arr_lang):
        layer_arr = layer_arr_2
        df["layer"] = layer_arr
    else:
        df["layer"] = layer_arr_lang
    if len(acc_arr_lang) < len(acc_arr):
        while len(acc_arr_lang) < len(acc_arr):
            acc_arr_lang.append(0)
            acc_auc_lang.append(0)
    if len(acc_arr) < len(acc_arr_lang):
        while len(acc_arr) < len(acc_arr_lang):
            acc_arr.append(0)
            acc_auc.append(0)

    print(len(acc_arr_lang))
    print(len(acc_auc_lang))
    print(len(acc_arr))
    print(len(acc_auc))
    df["acc_lang"] = acc_arr_lang
    df["auc_lang"] = acc_auc_lang
    df["acc"] = acc_arr
    df["auc"] = acc_auc
    model_name = args.model.replace("/","")
    df.to_csv(
                f"./results/{model_name}/prueba{args.round}test{args.samples}_buena_no_scale_{args.dataset}_aoooo.csv",
                index=False,
            )
