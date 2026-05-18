import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from lighthouse.common.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


class LengthWiseHungarianMatcher(nn.Module):

    def __init__(
        self,
        num_classes=3,
        boundaries=None,
        cost_span: float = 1,
        cost_giou: float = 1,
        cost_class: float = 1,
        span_loss_type: str = "l1",
        max_v_l: int = 75,
    ):

        super().__init__()
        self.num_classes = num_classes
        self.boundaries = boundaries if boundaries else [0.05, 0.2]

        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.cost_class = cost_class
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert (
            cost_class != 0 or cost_span != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_spans"].shape[:2]

        assert num_queries % self.num_classes == 0

        queries_per_class = num_queries // self.num_classes

        targets = targets["span_labels"]

        out_prob = outputs["pred_logits"].softmax(-1)
        out_spans = outputs["pred_spans"]

        boundaries_tensor = torch.tensor(self.boundaries, device=out_spans.device)

        indices = []

        for b in range(bs):

            pred_prob_b = out_prob[b]
            pred_spans_b = out_spans[b]

            tgt_spans = targets[b]["spans"]

            if len(tgt_spans) == 0:

                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )

                continue

            tgt_lengths = tgt_spans[:, 1]

            tgt_classes = torch.bucketize(tgt_lengths, boundaries_tensor)

            batch_src_idx = []
            batch_tgt_idx = []

            for k in range(self.num_classes):

                # -----------------------------
                # query group
                # -----------------------------
                q_start = k * queries_per_class
                q_end = (k + 1) * queries_per_class

                pred_spans_k = pred_spans_b[q_start:q_end]
                pred_prob_k = pred_prob_b[q_start:q_end]

                # -----------------------------
                # gt group
                # -----------------------------
                tgt_mask = tgt_classes == k

                if tgt_mask.sum() == 0:
                    continue

                tgt_spans_k = tgt_spans[tgt_mask]

                tgt_idx_global = torch.where(tgt_mask)[0]

                # -----------------------------
                # costs
                # -----------------------------
                cost_class = -pred_prob_k[:, self.foreground_label].unsqueeze(1)

                cost_span = torch.cdist(pred_spans_k, tgt_spans_k, p=1)

                cost_giou = -generalized_temporal_iou(
                    span_cxw_to_xx(pred_spans_k), span_cxw_to_xx(tgt_spans_k)
                )

                C = (
                    self.cost_class * cost_class
                    + self.cost_span * cost_span
                    + self.cost_giou * cost_giou
                )

                C = C.cpu()

                src_idx, tgt_idx_local = linear_sum_assignment(C)

                src_idx = torch.as_tensor(src_idx, dtype=torch.int64)

                tgt_idx_local = torch.as_tensor(tgt_idx_local, dtype=torch.int64)

                src_idx += q_start

                tgt_idx = tgt_idx_global[tgt_idx_local]

                batch_src_idx.append(src_idx)
                batch_tgt_idx.append(tgt_idx)

            if len(batch_src_idx) == 0:

                batch_src_idx = torch.empty(0, dtype=torch.int64)

                batch_tgt_idx = torch.empty(0, dtype=torch.int64)

            else:

                batch_src_idx = torch.cat(batch_src_idx)
                batch_tgt_idx = torch.cat(batch_tgt_idx)

            indices.append((batch_src_idx, batch_tgt_idx))

        return indices


def build_matcher(args):
    # Передаем num_classes и boundaries из args, если они есть, или дефолтные
    num_classes = getattr(args, "lad_num_classes", 3)
    # Если границы не заданы в args, можно передать None (возьмет [0.2, 0.5])
    boundaries = getattr(args, "lad_boundaries", None)

    return LengthWiseHungarianMatcher(
        num_classes=num_classes,
        boundaries=boundaries,
        cost_span=args.set_cost_span,
        cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
    )


def build_event_matcher(args):
    # Передаем num_classes и boundaries из args, если они есть, или дефолтные
    num_classes = getattr(args, "lad_num_classes", 3)
    # Если границы не заданы в args, можно передать None (возьмет [0.2, 0.5])
    boundaries = getattr(args, "lad_boundaries", None)

    return LengthWiseHungarianMatcher(
        num_classes=num_classes,
        boundaries=boundaries,
        cost_span=args.set_cost_span,
        cost_giou=args.set_cost_giou,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
    )
