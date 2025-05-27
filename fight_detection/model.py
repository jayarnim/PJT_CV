import torch
import torch.nn as nn


class FightDetection(nn.Module):
    def __init__(
        self, 
        hidden: list=[400, 256, 128, 64],
        dropout: float=0.2,
    ):
        super(FightDetection, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.hidden = hidden
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._init_layers()

    def forward(self, X):
        return self._score(X)

    def predict(self, X):
        with torch.no_grad():
            logit = self._score(X)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, X):
        extracted_feature = self.feature_extractor(X)
        logit = self.classifier(extracted_feature).squeeze(-1)
        return logit

    def _init_layers(self):
        self._feature_extractor_generator()
        self._classifier_generator()

    def _feature_extractor_generator(self):
        self.pretrained_model = torch.hub.load(
            repo_or_dir='facebookresearch/pytorchvideo',
            model='slowfast_r50',
            pretrained=True,
        )
        self.pretrained_model = self.pretrained_model.eval()

        self.feature_extractor = nn.Sequential(
            *list(self.pretrained_model.blocks[:])
        )

        for param in self.feature_extractor[:5].parameters():
            param.requires_grad = False

    def _classifier_generator(self):
        self.classifier = nn.Sequential(
            *list(self._generate_layers(self.hidden)),
            nn.Linear(self.hidden[-1], 1)
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == 400)
        ERROR_MESSAGE = f"First MLP input dim must be 400, but got {self.hidden[0]}"
        assert CONDITION, ERROR_MESSAGE
