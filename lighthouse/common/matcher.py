import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from lighthouse.common.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


class LengthWiseHungarianMatcher(nn.Module):
    """
    Length-Aware Matcher.
    Разбивает queries и targets на группы по длине и делает matching внутри групп.
    """

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
        """
        Args:
            num_classes: Количество групп длины (например, 3: short, mid, long)
            boundaries: Границы длин. Например [0.2, 0.5].
                        Если None, использует дефолтные.
        """
        super().__init__()
        self.num_classes = num_classes
        # Границы: < 0.2 -> class 0, < 0.5 -> class 1, >= 0.5 -> class 2
        self.boundaries = boundaries if boundaries else [10, 30]

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
        targets = targets["span_labels"]
        
        out_spans = outputs["pred_spans"]          # [bs, Q, 2]
        out_prob = outputs["pred_logits"].softmax(-1)  # [bs, Q, num_classes+1]?

        boundaries_tensor = torch.tensor(self.boundaries, device=out_spans.device)
        
        # Классы запросов (одинаковы для всех элементов батча)
        queries_per_class = num_queries // self.num_classes
        query_classes = torch.arange(num_queries, device=out_spans.device) // queries_per_class  # [Q]

        indices = []
        for i in range(bs):
            tgt_spans_i = targets[i]["spans"]      # [n_tgt, 2]
            n_tgt = len(tgt_spans_i)
            
            # Матрица стоимости [Q, n_tgt + 1] (последний столбец – dummy)
            C_i = torch.full((num_queries, n_tgt + 1), float("inf"), device=out_spans.device)
            
            if n_tgt > 0:
                tgt_lengths_i = tgt_spans_i[:, 1]
                tgt_classes_i = torch.bucketize(tgt_lengths_i, boundaries_tensor)
                tgt_ids_i = torch.full((n_tgt,), self.foreground_label, device=out_spans.device)
                
                pred_spans_i = out_spans[i]  # [Q, 2]
                prob_i = out_prob[i]         # [Q, num_classes]
                
                for k in range(self.num_classes):
                    q_mask = query_classes == k
                    t_mask = tgt_classes_i == k
                    q_idx_k = torch.where(q_mask)[0]
                    t_idx_k = torch.where(t_mask)[0]
                    if len(t_idx_k) == 0:
                        continue
                    preds_k = pred_spans_i[q_idx_k]
                    tgts_k = tgt_spans_i[t_idx_k]
                    prob_k = prob_i[q_idx_k]
                    cost_class = -prob_k[:, tgt_ids_i[t_idx_k]]
                    cost_span = torch.cdist(preds_k, tgts_k, p=1)
                    cost_giou = -generalized_temporal_iou(
                        span_cxw_to_xx(preds_k), span_cxw_to_xx(tgts_k)
                    )
                    C_k = (self.cost_span * cost_span +
                        self.cost_giou * cost_giou +
                        self.cost_class * cost_class)
                    C_i[q_idx_k[:, None], t_idx_k] = C_k
            
            # Стоимость для dummy-цели
            if out_prob.shape[-1] > self.foreground_label + 1:
                # предполагаем, что последний класс – «нет объекта»
                cost_no_match = -out_prob[i, :, -1]   # [Q]
            else:
                cost_no_match = torch.full((num_queries,), 10.0, device=out_spans.device)
            C_i[:, n_tgt] = cost_no_match
            
            # Венгерский алгоритм
            row, col = linear_sum_assignment(C_i.cpu())
            row = torch.as_tensor(row, dtype=torch.int64)
            col = torch.as_tensor(col, dtype=torch.int64)
            mask = col != n_tgt            # отбрасываем dummy-назначения
            indices.append((row[mask], col[mask]))
        
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
