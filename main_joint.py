# -*- coding: utf-8 -*-
import os
import random
import warnings

from data_util.data_process import *
from tqdm import tqdm, trange
from data_util.Metrics import IntentMetrics, SlotMetrics,semantic_acc
from model.joint_model_trans import Joint_model
from model.Radam import RAdam


warnings.filterwarnings('ignore')
if config.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda", torch.cuda.current_device())
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False


def set_seed():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not config.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def dev(model, dev_loader, idx2slot):

    model.eval()
    eval_loss_intent = 0
    eval_loss_slot = 0
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []
    total_wrong = 0
    
    for i, batch in enumerate(tqdm(dev_loader, desc="Evaluating")):
        inputs, char_lists, slot_labels, intent_labels, masks, = batch
        if use_cuda:
            inputs, char_lists, masks, intent_labels, slot_labels = \
                inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
        loss_intent, loss_slot = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

        pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
        pred_intents.extend(pred_intent.cpu().numpy().tolist())
        true_intents.extend(intent_labels.cpu().numpy().tolist())
        eval_loss_intent += loss_intent.item()
        eval_loss_slot += loss_slot.item()
        slot_labels = slot_labels.cpu().numpy().tolist()

        for idx_in_batch in range(len(pred_slot)):
            pred = []
            true = []
            # Get full slot lists (including <start>/<end>)
            for j in range(len(pred_slot[idx_in_batch])):
                pred.append(idx2slot[pred_slot[idx_in_batch][j].item()])
                true.append(idx2slot[batch_slot_labels[idx_in_batch][j]])
            
            # Add to metric lists (stripping <start>/<end>)
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])

            # --- ADDED: Error checking and logging ---
            pred_intent_id = batch_pred_intents[idx_in_batch]
            true_intent_id = batch_true_intents[idx_in_batch]

            intent_is_wrong = (pred_intent_id != true_intent_id)
            slots_are_wrong = (pred != true) # Check full list

            if intent_is_wrong or slots_are_wrong:
                total_wrong += 1
                results_writer.write(f"Example {total_wrong}\n")
                
                input_ids_list = inputs[idx_in_batch].cpu().numpy().tolist()
                results_writer.write(f"Input IDs: {input_ids_list}\n")
                
                pred_intent_str = idx2intent[pred_intent_id]
                true_intent_str = idx2intent[true_intent_id]
                results_writer.write(f"Intent Pred: {pred_intent_str} (True: {true_intent_str})\n")
                
                pred_slot_str = " ".join(pred)
                true_slot_str = " ".join(true)
                results_writer.write(f"Slot Pred: {pred_slot_str}\n")
                results_writer.write(f"Slot True: {true_slot_str}\n")
                results_writer.write("-" * 40 + "\n")
    # slot f1, p, r
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, slot_p, slot_r = slot_metrics.get_slot_metrics()
    # intent f1, p, r
    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    intent_acc = Metrics_intent.accuracy
    data_nums = len(dev_loader.dataset)
    ave_loss_intent = eval_loss_intent * config.batch_size / data_nums
    ave_loss_slot = eval_loss_slot * config.batch_size / data_nums

    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation - intent_loss: {:.6f} slot_loss: {:.6f} acc: {:.4f}% '
          'slot f1: {:.4f} sent acc: {:.4f} \n'.format(ave_loss_intent, ave_loss_slot,
                                                       intent_acc, slot_f1, sent_acc))
    model.train()

    return intent_acc, slot_f1, sent_acc, ave_loss_intent, ave_loss_slot, total_wrong


def run_train(train_data_file, dev_data_file):

    print("1. load config and dict")
    vocab_file = open(config.data_path + "vocab.txt", "r", encoding="utf-8")
    vocab_list = [word.strip() for word in vocab_file]
    if not os.path.exists(config.data_path + "emb_word.txt"):
        emb_file = "/kaggle/input/w2v-vn-word-level/word2vec_vi_words_300dims.txt"
        embeddings = read_emb(emb_file, vocab_list)
        emb_write = open(config.data_path + "/emb_word.txt", "w", encoding="utf-8")
        for emb in embeddings:
            emb_write.write(emb)
        emb_write.close()
    else:
        embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
        embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    idx2intent, intent2idx = lord_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = lord_label_dict(config.data_path + "slot_label.txt")
    n_slot_tag = len(idx2slot.items())
    n_intent_class = len(idx2intent.items())

    train_dir = os.path.join(config.data_path, train_data_file)
    dev_dir = os.path.join(config.data_path, dev_data_file)
    train_loader = read_corpus(train_dir, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                               vocab=vocab, is_train=True)
    dev_loader = read_corpus(dev_dir, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                             vocab=vocab, is_train=False)
    model = Joint_model(config, config.hidden_dim, config.batch_size, config.max_len, n_intent_class, n_slot_tag, embedding_word)

    if use_cuda:
        model.cuda()
    model.train()
    optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=0.000001)
    best_slot_f1 = [0.0, 0.0, 0.0]
    best_intent_acc = [0.0, 0.0, 0.0]
    best_sent_acc = [0.0, 0.0, 0.0]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 70], gamma=config.lr_scheduler_gama, last_epoch=-1)

    for epoch in trange(config.epoch, desc="Epoch"):
        print(scheduler.get_lr())
        step = 0
        for i, batch in enumerate(tqdm(train_loader, desc="batch_nums")):
            step += 1
            model.zero_grad()
            inputs, char_lists, slot_labels, intent_labels, masks, = batch
            if use_cuda:
                inputs, char_lists, masks, intent_labels, slot_labels = \
                    inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
            logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            loss_intent, loss_slot, = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

            if epoch < 40:
                loss = loss_slot + loss_intent
            else:
                loss = 0.8 * loss_intent + 0.2 * loss_slot
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("loss domain:", loss.item())
                print('epoch: {}|    step: {} |    loss: {}'.format(epoch, step, loss.item()))

        dev_results_file_path = os.path.join(config.model_save_dir, f"epoch_{epoch}_dev_wrong_predictions.txt")
        total_dev_wrong = 0
        
        with open(dev_results_file_path, 'w', encoding='utf-8') as f_dev_wrong:
            f_dev_wrong.write(f"***** Wrong Predictions (Dev Set) - Epoch {epoch} *****\n\n")
            f_dev_wrong.write("========================================\n\n")

            # Call dev with new signature, passing the file handle and idx2intent
            (intent_acc, slot_f1, sent_acc, 
             ave_loss_intent, ave_loss_slot, 
             total_dev_wrong) = dev(model, dev_loader, idx2slot, idx2intent, f_dev_wrong)
            
            # Write footer with total error count
            f_dev_wrong.write(f"\n========================================\n")
            f_dev_wrong.write(f"Total wrong examples: {total_dev_wrong}\n")


        with open(results_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch}\t{current_lr:.1E}\t{ave_loss_intent:.6f}\t{ave_loss_slot:.6f}\t{intent_acc:.4f}\t{slot_f1:.4f}\t{sent_acc:.4f}\n")

        if slot_f1 > best_slot_f1[1] :
            best_slot_f1 = [sent_acc, slot_f1, intent_acc, epoch]
            torch.save(model, config.model_save_dir + config.model_path)
        if intent_acc > best_intent_acc[2]:
            torch.save(model, config.model_save_dir + config.model_path)
            best_intent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        if sent_acc > best_sent_acc[0]:
            torch.save(model, config.model_save_dir + config.model_path)
            best_sent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        scheduler.step()
    print("best_slot_f1:", best_slot_f1)
    print("best_intent_acc:", best_intent_acc)
    print("best_sent_acc:", best_sent_acc)


def run_test(test_data_file):
    # load dict
    idx2intent, intent2idx = lord_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = lord_label_dict(config.data_path + "slot_label.txt")

    embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    test_dir = os.path.join(config.data_path, test_data_file)
    test_loader = read_corpus(test_dir, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                              vocab=vocab, is_train=False)
    model = torch.load(config.model_save_dir + config.model_path, map_location=device, weights_only=False)
    model.eval()

    # --- MODIFICATION: Create test error log file ---
    test_results_file_path = os.path.join(config.model_save_dir, "test_wrong_predictions.txt")
    total_test_wrong = 0
    
    with open(test_results_file_path, 'w', encoding='utf-8') as f_test_wrong:
        f_test_wrong.write("***** Wrong Predictions (Test Set) *****\n\n")
        f_test_wrong.write("========================================\n\n")

        # Lists for overall metrics
        pred_intents = []
        true_intents = []
        pred_slots = []
        true_slots = []

        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, char_lists, slot_labels, intent_labels, masks, = batch
            if use_cuda:
                inputs, char_lists, masks, intent_labels, slot_labels = inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
            
            logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
            
            # Get batch lists for error checking
            batch_pred_intents = pred_intent.cpu().numpy().tolist()
            batch_true_intents = intent_labels.cpu().numpy().tolist()
            batch_slot_labels = slot_labels.cpu().numpy().tolist()

            # Add to overall lists for metrics
            pred_intents.extend(batch_pred_intents)
            true_intents.extend(batch_true_intents)

            # --- MODIFIED: Fixed 'i' shadowing bug and added error logging ---
            # Use 'idx_in_batch' instead of 'i' to avoid conflict
            for idx_in_batch in range(len(pred_slot)):
                pred = []
                true = []
                # Get full slot lists (including <start>, <end>, <PAD>)
                for j in range(len(pred_slot[idx_in_batch])):
                    pred.append(idx2slot[pred_slot[idx_in_batch][j].item()])
                    true.append(idx2slot[batch_slot_labels[idx_in_batch][j]])
                
                # Add to metric lists (stripping <start>/<end>)
                pred_slots.append(pred[1:-1])
                true_slots.append(true[1:-1])

                # --- ADDED: Error checking and logging ---
                pred_intent_id = batch_pred_intents[idx_in_batch]
                true_intent_id = batch_true_intents[idx_in_batch]

                intent_is_wrong = (pred_intent_id != true_intent_id)
                # Check full list (including special tokens) for mismatches
                slots_are_wrong = (pred != true) 

                if intent_is_wrong or slots_are_wrong:
                    total_test_wrong += 1
                    f_test_wrong.write(f"Example {total_test_wrong}\n")
                    
                    input_ids_list = inputs[idx_in_batch].cpu().numpy().tolist()
                    f_test_wrong.write(f"Input IDs: {input_ids_list}\n")
                    
                    pred_intent_str = idx2intent[pred_intent_id]
                    true_intent_str = idx2intent[true_intent_id]
                    f_test_wrong.write(f"Intent Pred: {pred_intent_str} (True: {true_intent_str})\n")
                    
                    pred_slot_str = " ".join(pred)
                    true_slot_str = " ".join(true)
                    f_test_wrong.write(f"Slot Pred: {pred_slot_str}\n")
                    f_test_wrong.write(f"Slot True: {true_slot_str}\n")
                    f_test_wrong.write("-" * 40 + "\n")
                # --- END ADDED ---
        
        # --- ADDED: Write footer ---
        f_test_wrong.write(f"\n========================================\n")
        f_test_wrong.write(f"Total wrong examples: {total_test_wrong}\n")
    
    # --- END 'with open' block ---

    # slot f1, p, r
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, _, _ = slot_metrics.get_slot_metrics()

    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    print(Metrics_intent.classification_report)
    intent_acc = Metrics_intent.accuracy
    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation -  acc: {:.4f}% ' 'slot f1: {:.4f} sent_acc: {:.4f}  \n'.format(intent_acc, slot_f1, sent_acc))

    # --- MODIFICATION: Write test results to main training file ---
    # (This part was from the previous step, make sure it's here)
    results_file_path = os.path.join(config.model_save_dir, "training_results.txt")
    with open(results_file_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 30 + "\n")
        f.write(f"--- FINAL TEST RESULTS ---\n")
        f.write(f"Model: {config.model_path}\n")
        f.write(f"Test File: {test_data_file}\n")
        f.write(f"Test Intent Accuracy: {intent_acc:.4f}\n")
        f.write(f"Test Slot F1: {slot_f1:.4f}\n")
        f.write(f"Test Sentence Accuracy: {sent_acc:.4f}\n\n")
        f.write("--- Test Intent Classification Report ---\n")
        f.write(Metrics_intent.classification_report + "\n")
    # --- END MODIFICATION ---

    return sent_acc


if __name__ == "__main__":
    train_file = "train.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"
    #trian model
    set_seed()
    run_train(train_file, dev_file)
    #test model
    run_test(test_file)
