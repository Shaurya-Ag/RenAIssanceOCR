import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import editdistance

class CRNNTrainer:
    def __init__(self, model, train_loader, val_loader, blank_idx, idx_to_label, device, IMG_MEAN = 0.8477863073348999, IMG_STD = 0.2494666874408722):
        self.IMG_MEAN = IMG_MEAN
        self.IMG_STD = IMG_STD
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ctc_loss = nn.CTCLoss(blank=blank_idx,zero_infinity=True)
        self.blank_idx = blank_idx
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.idx_to_label = idx_to_label
        self.device = device
        
        self.writer = SummaryWriter(log_dir="./runs/ocr_crnn")
        self.activations = {}
        model.conv1.register_forward_hook(self.get_activation("conv1"))
        model.conv2.register_forward_hook(self.get_activation("conv2"))
        model.conv3.register_forward_hook(self.get_activation("conv3"))
        model.conv4.register_forward_hook(self.get_activation("conv4"))
        model.embedding.register_forward_hook(self.get_activation("embedding"))
        model.lstm.register_forward_hook(self.get_activation("lstm"))
    
    def train_step(self, batch):
        images, targets, input_lengths, target_lengths = batch

        images = images.to(self.device)
        targets = targets.to(self.device)
        input_lengths = input_lengths.to(self.device)
        target_lengths = target_lengths.to(self.device)

        self.optimizer.zero_grad()

        log_probs = self.model(images)
        # shape: [T, B, C]

        loss = self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()
            
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            loss = self.train_step(batch)
            total_loss += loss

        return total_loss / len(self.train_loader)
    
    def train(self, num_epochs):
        global_step = 0
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
                self.model.train()

                for batch in self.train_loader:
                    images, targets, input_lengths, target_lengths = batch

                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    input_lengths = input_lengths.to(self.device)
                    target_lengths = target_lengths.to(self.device)

                    self.optimizer.zero_grad()

                    log_probs = self.model(images)

                    loss = self.ctc_loss(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths
                    )

                    loss.backward()

                    # optional but I'll be using it
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                    #  log loss
                    self.writer.add_scalar("loss/train", loss.item(), global_step)

                    # occasional heavy logs
                    if global_step % 10 == 0:
                        self.log_gradients(self.model, global_step)
                        self.log_activations(global_step)
                        self.log_predictions(images, log_probs, targets, target_lengths, global_step)

                    #  quick validation
                    if global_step % 20 == 0:
                        val_losses.append((global_step, self.validate(global_step)))
                    self.writer.flush()
                    global_step += 1
                    train_losses.append((global_step, loss.item()))

                print(f"Epoch {epoch+1} done")
        return train_losses, val_losses
    def log_predictions(self, images, log_probs, targets, target_lengths, step):
        targets = targets.detach().cpu()
        images = images.detach().cpu()  # unnormalize for visualization
        preds = self.greedy_decode(log_probs.detach().cpu())

        # reconstruct ground truth
        gt = []
        idx = 0
        for l in target_lengths:
            gt.append(targets[idx:idx+l].tolist())
            idx += l

        for i in range(min(4, len(preds))):
            pred_str = self.decode_to_string(preds[i])
            gt_str = self.decode_to_string(gt[i])
            self.writer.add_text(f"predictions/{i}",f"GT: {gt_str} | Pred: {pred_str}",step)
            images[i] = (images[i] * self.IMG_STD) + self.IMG_MEAN
            self.writer.add_image(f"image/{i}", images[i], step)
            
    def decode_to_string(self, seq):
        return "".join([self.idx_to_label[i] for i in seq])
    
    def greedy_decode(self, log_probs):
        # log_probs: [T, B, C]
        preds = log_probs.argmax(2)  # [T, B]

        preds = preds.permute(1, 0)  # [B, T]

        decoded = []

        for seq in preds:
            prev = None
            out = []
            for p in seq:
                p = p.item()
                if p != self.blank_idx and p != prev:
                    out.append(p)
                prev = p
            decoded.append(out)

        return decoded

    def decode_predictions(self, preds):
        # preds: [T, B, C]
        preds = preds.permute(1, 0, 2)  # [B, T, C]
        pred_labels = torch.argmax(preds, dim=2)  # [B, T]
        
        decoded_strings = []
        for seq in pred_labels:
            prev_idx = self.blank_idx  # Start with blank
            decoded_str = ""
            for idx in seq:
                idx = idx.item()
                if idx != prev_idx and idx != self.blank_idx:
                    decoded_str += self.idx_to_label[idx]
                prev_idx = idx
            decoded_strings.append(decoded_str)
        
        return decoded_strings
    
    def log_gradients(self, model, step):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, step)
                
    def get_activation(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook
    
    def log_activations(self, step):
        for name, act in self.activations.items():
            self.writer.add_histogram(f"activations/{name}", act, step)
    
    @torch.no_grad()
    def quick_validate(self, val_loader):
        val_iter = iter(val_loader)
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)

        return batch
    
    @torch.no_grad()
    def validate(self, global_step):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        for batch in self.val_loader:
            images, targets, input_lengths, target_lengths = batch

            images = images.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)

            log_probs = self.model(images)
            
            
            preds = self.decode_predictions(log_probs)
            all_preds.extend(preds)
            idx = 0
            for l in target_lengths:
                gt_seq = targets[idx:idx+l].tolist()
                gt_str = "".join([self.idx_to_label[i] for i in gt_seq])
                all_targets.append(gt_str)
                idx += l

            loss = self.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

            total_loss += loss.item()
        self.writer.add_scalar("loss/val", total_loss / len(self.val_loader), global_step)
        self.model.train()
        wer = self.calculate_wer(all_preds, all_targets)
        cer = self.calculate_cer(all_preds, all_targets)
        self.writer.add_scalar("metrics/val_cer", cer, global_step)
        self.writer.add_scalar("metrics/val_wer", wer, global_step)
        return total_loss / len(self.val_loader)
    @torch.no_grad()
    def final_validation(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            images, targets, input_lengths, target_lengths = batch

            images = images.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)

            log_probs = self.model(images)

            loss = self.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

            total_loss += loss.item()

            preds = self.decode_predictions(log_probs)
            all_preds.extend(preds)

            idx = 0
            for l in target_lengths:
                gt_seq = targets[idx:idx+l].tolist()
                gt_str = "".join([self.idx_to_label[i] for i in gt_seq])
                all_targets.append(gt_str)
                idx += l

        avg_loss = total_loss / len(self.val_loader)
        wer = self.calculate_wer(all_preds, all_targets)
        cer = self.calculate_cer(all_preds, all_targets)
        acc = self.accuracy(all_preds, all_targets)
        return avg_loss, wer, cer, acc, all_preds[:16], all_targets[:16]
        
    def calculate_wer(self, preds, targets):
        # right now, it is equivalent to word accuracy because each input is a single word, 
        # but this will be useful when we move to line-level recognition
        total_errors = 0
        total_words = 0
        for pred, target in zip(preds, targets):
            pred_words = pred.split()
            target_words = target.split()
            total_errors += editdistance.eval(pred_words, target_words)
            total_words += len(target_words)
        return total_errors / total_words if total_words > 0 else 0
    
    def calculate_cer(self, preds, targets):
        total_errors = 0
        total_chars = 0
        for pred, target in zip(preds, targets):
            total_errors += editdistance.eval(pred, target)
            total_chars += len(target)
        return total_errors / total_chars if total_chars > 0 else 0
    
    def accuracy(self, preds, targets):
        correct = 0
        total = 0
        for pred, target in zip(preds, targets):
            for l, t in zip(pred, target):
                if l == t:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.