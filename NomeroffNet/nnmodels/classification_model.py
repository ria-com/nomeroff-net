import pytorch_lightning as pl


class ClassificationNet(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log(f'Batch {batch_idx} train_loss', loss)
        self.log(f'Batch {batch_idx} accuracy', acc)
        return {
            'loss': loss,
            'acc': acc,
        }

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('val_loss', loss)
        self.log(f'val_accuracy', acc)
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('test_loss', loss)
        self.log(f'test_accuracy', acc)
        return {
            'test_loss': loss,
            'test_acc': acc,
        }
