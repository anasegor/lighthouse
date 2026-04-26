"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Copyright (c) 2022 WonJun Moon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from lighthouse.common import vocab
from lighthouse.common.utils.basic_utils import load_jsonl, l2_normalize_np_array
from lighthouse.common.utils.tensor_utils import pad_sequences_1d
from lighthouse.common.utils.span_utils import span_xx_to_cxw
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class AudioMomentMix(nn.Module):
    def __init__(
        self,
        epsilon_cut: float = 5.0,      # длина под‑сегмента в секундах
        prob: float = 0.5,
        use_background_mix: bool = True,
        min_moment_length: int = 1,
        clip_len: float = 1.0          # длительность одного аудио‑шага в секундах
    ):
        super().__init__()
        self.epsilon_cut = epsilon_cut
        self.prob = prob
        self.use_background_mix = use_background_mix
        self.min_moment_length = min_moment_length
        self.clip_len = clip_len

    def forward(
        self,
        audio_features: torch.Tensor,  # [T, D]
        text_features: torch.Tensor,   # [D']
        moment_start: int,
        moment_end: int,
        video_id: str,
        all_videos_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if torch.rand(1).item() > self.prob:
            return audio_features, text_features, moment_start, moment_end

        # Stage 1: ForegroundMix
        audio_fg_mix, fg_ranges = self._foreground_mix(
            audio_features, moment_start, moment_end
        )

        # Stage 2: BackgroundMix
        if self.use_background_mix and all_videos_data is not None and len(all_videos_data) > 1:
            audio_final = self._background_mix(
                audio_fg_mix, fg_ranges, video_id, all_videos_data
            )
            return audio_final, text_features, fg_ranges
        else:
            return audio_fg_mix, text_features, fg_ranges

    def _foreground_mix(
        self,
        audio_features: torch.Tensor,
        moment_start: int,
        moment_end: int
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        T, D = audio_features.shape
        fg_length = moment_end - moment_start
        if fg_length < self.min_moment_length:
            return audio_features, [(moment_start, moment_end)]

        # Переводим epsilon_cut в количество индексов
        epsilon_idx = max(1, int(self.epsilon_cut / self.clip_len))
        n_subsegments = max(1, int(fg_length / epsilon_idx))
        if n_subsegments <= 1:
            return audio_features, [(moment_start, moment_end)]

        # Делим foreground на подсегменты
        fg_features = audio_features[moment_start:moment_end]
        fg_subsegments = list(torch.chunk(fg_features, n_subsegments, dim=0))
        fg_subsegments = [seg for seg in fg_subsegments if seg.shape[0] > 0]
        n_fg = len(fg_subsegments)

        # Background = до + после момента
        bg_before = audio_features[:moment_start]
        bg_after = audio_features[moment_end:]
        bg_features = torch.cat([bg_before, bg_after], dim=0)
        bg_subsegments = list(torch.chunk(bg_features, n_fg + 1, dim=0))
        bg_subsegments = [seg for seg in bg_subsegments if seg.shape[0] > 0]
        # дополняем до нужного количества
        while len(bg_subsegments) < n_fg + 1:
            bg_subsegments.append(torch.zeros(0, D, dtype=audio_features.dtype, device=audio_features.device))
        bg_subsegments = bg_subsegments[:n_fg + 1]

        # Перемешиваем
        fg_perm = torch.randperm(n_fg)
        bg_perm = torch.randperm(n_fg + 1)
        fg_shuffled = [fg_subsegments[i] for i in fg_perm]
        bg_shuffled = [bg_subsegments[i] for i in bg_perm]

        # Собираем: b'_0, затем пары (f'_i, b'_{i+1}) для всех fg, хвост из оставшихся bg
        mixed = [bg_shuffled[0]]
        for i in range(n_fg):
            mixed.append(fg_shuffled[i])
            if i + 1 < len(bg_shuffled):
                mixed.append(bg_shuffled[i + 1])
        for j in range(n_fg + 1, len(bg_shuffled)):
            mixed.append(bg_shuffled[j])

        audio_mixed = torch.cat(mixed, dim=0)

        # Вычисляем абсолютные позиции каждого fg‑сегмента в новой последовательности
        if audio_mixed.shape[0] < T:
            deficit = T - audio_mixed.shape[0]
            if bg_features.shape[0] >= deficit:
                extra_bg = bg_features[-deficit:]
            else:
                repeats = deficit // bg_features.shape[0] + 1
                extra_bg = bg_features.repeat(repeats, 1)[:deficit]
            audio_mixed = torch.cat([audio_mixed, extra_bg], dim=0)
        elif audio_mixed.shape[0] > T:
            audio_mixed = audio_mixed[:T]

        offset = 0
        fg_ranges = []
        for seg in mixed:
            if seg in fg_shuffled:
                fg_ranges.append((offset, offset + seg.shape[0]))
            offset += seg.shape[0]

        # обрезаем диапазоны, выходящие за T (если был padding)
        fg_ranges = [(s, e) for (s, e) in fg_ranges if s < T]
        fg_ranges = [(min(s, T), min(e, T)) for (s, e) in fg_ranges]

        return audio_mixed, fg_ranges

    def _background_mix(
        self,
        audio_features: torch.Tensor,    # [T, D]
        fg_ranges: List[Tuple[int, int]],# список (start, end) – границы foreground
        video_id: str,
        all_videos_data: Dict[str, torch.Tensor]   # {vid: tensor}
    ) -> torch.Tensor:
        """
        Заменяет все участки аудио, НЕ входящие ни в один из fg_ranges,
        на фрагменты из случайно выбранного другого видео.
        """
        T, D = audio_features.shape
        other_videos = [vid for vid in all_videos_data.keys() if vid != video_id]
        if not other_videos:
            return audio_features

        # создаём маску: True – foreground, False – фон
        fg_mask = torch.zeros(T, dtype=torch.bool, device=audio_features.device)
        for s, e in fg_ranges:
            if s < e:
                fg_mask[s:min(e, T)] = True

        # выбираем одно другое видео
        other_vid = random.choice(other_videos)
        other_audio = all_videos_data[other_vid]  # [T_other, D]
        T_other = other_audio.shape[0]

        audio_bgmix = audio_features.clone()

        # проходим по всем непрерывным фоновым участкам
        bg_intervals = []
        in_bg = False
        start = 0
        for i in range(T):
            if not fg_mask[i] and not in_bg:
                start = i
                in_bg = True
            elif fg_mask[i] and in_bg:
                bg_intervals.append((start, i))
                in_bg = False
        if in_bg:
            bg_intervals.append((start, T))

        # заменяем каждый фоновый интервал случайным куском из other_audio
        for bg_start, bg_end in bg_intervals:
            seg_len = bg_end - bg_start
            if seg_len <= 0:
                continue
            if T_other > seg_len:
                crop_start = torch.randint(0, T_other - seg_len + 1, (1,)).item()
                cropped = other_audio[crop_start:crop_start + seg_len]
            else:
                cropped = other_audio
                if cropped.shape[0] < seg_len:
                    # повторяем, чтобы заполнить нужную длину
                    repeats = seg_len // cropped.shape[0] + 1
                    cropped = cropped.repeat(repeats, 1)[:seg_len]
            audio_bgmix[bg_start:bg_end] = cropped.to(audio_bgmix.device)

        return audio_bgmix
    
class Castella_StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(
        self,
        dset_name,
        domain,
        data_path,
        v_feat_dirs,
        a_feat_dirs,
        q_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types="clip",
        a_feat_types="pann",
        max_q_l=32,
        max_v_l=75,
        max_a_l=75,
        ctx_mode="audio",
        clip_len=2,
        max_windows=5,
        span_loss_type="l1",
        load_labels=True,

        use_moment_mix=True,
        moment_mix_epsilon=5.0,
        moment_mix_prob=0.5,
        moment_mix_bg=True,    
        moment_mix_min_len=1,

    ):
        self.dset_name = dset_name
        self.domain = domain
        self.data_path = data_path
        self.v_feat_dirs = (
            v_feat_dirs if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        )
        self.a_feat_dirs = (
            a_feat_dirs if isinstance(a_feat_dirs, list) else [a_feat_dirs]
        )
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_types = v_feat_types
        self.a_feat_types = a_feat_types

        if max_v_l == -1:
            max_v_l = 100000000
        if max_a_l == -1:
            max_a_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.max_a_l = max_a_l

        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels


         # ==== MomentMix integration ====
        self.use_moment_mix = use_moment_mix
        if self.use_moment_mix:
            self.moment_mix = AudioMomentMix(
                epsilon_cut=moment_mix_epsilon,
                prob=moment_mix_prob,
                use_background_mix=moment_mix_bg,
                min_moment_length=moment_mix_min_len,
                clip_length=clip_len
            )
            self._all_audio_cache = None
        else:
            self.moment_mix = None

        # data
        self.data = self.load_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        return datalist

    def _load_all_audio_features(self):
        """Загружает аудиопризнаки всех уникальных видео в словарь {vid: tensor}"""
        if self._all_audio_cache is not None:
            return self._all_audio_cache
        cache = {}
        unique_vids = set(item["vid"] for item in self.data)
        for vid in unique_vids:
            a_feat = self._get_audio_feat_by_vid(vid)  # [L, D]
            cache[vid] = a_feat
        self._all_audio_cache = cache
        return cache
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(
            meta["qid"]
        )  # (Dq, ) or (Lq, Dq)

        ctx_l = self.max_v_l

        audio_feat = self._get_audio_feat_by_vid(meta["vid"])
        ctx_l_a = len(model_inputs["audio_feat"])
        # Sometimes, audio features is longer than video features because the length of video is not necessarily 2:30.
        if ctx_l < ctx_l_a:
            model_inputs["audio_feat"] = model_inputs["audio_feat"][:ctx_l]
            ctx_l_a = ctx_l
        elif ctx_l > ctx_l_a:
            ctx_l = ctx_l_a

        if self.use_moment_mix and len(meta.get("relevant_windows", [])) > 0:

            windows = meta["relevant_windows"]
            if len(windows) != 1:
                return
            # if len(windows) == 1:
            chosen_win = windows[0]
            # else:
                # chosen_win = random.choice(windows)
            st_sec, ed_sec = chosen_win
            clip_start = int(st_sec / self.clip_len)
            clip_end = int(ed_sec / self.clip_len)
            clip_start = max(0, min(clip_start, ctx_l - 1))
            clip_end = max(clip_start + 1, min(clip_end, ctx_l))  # хотя бы 1

            all_audio = self._load_all_audio_features() if self.moment_mix.use_background_mix else None

            # Применяем MomentMix
            audio_feat_mixed, _, fg_ranges = self.moment_mix(
                audio_feat,
                model_inputs["query_feat"],
                clip_start,
                clip_end,
                video_id=meta["vid"],
                all_videos_data=all_audio
            )
            audio_feat = audio_feat_mixed

            # fg_ranges: список [(start_idx, end_idx), ...] в индексах аудио (0..T-1)
            # Преобразуем обратно в секунды, формируя список окон
            new_windows = []
            for s, e in fg_ranges:
                new_windows.append([s * self.clip_len, e * self.clip_len])
            meta["relevant_windows"] = new_windows

        model_inputs["audio_feat"] = audio_feat

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(
            range(gt_ed + 1, ctx_l)
        )  # to fix bugs / works..?
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        score_array = np.zeros(ctx_l)
        score_array[gt_st : gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels(
        self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True
    ):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [
            min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]
        ]
        hard_neg_clip_indices = [
            min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]
        ]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(
        self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True
    ):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            # if rel_clip_ids[idx] == ctx_l:
            #     print(rel_clip_ids[idx], ctx_l)
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [
            min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]
        ]
        hard_neg_clip_indices = [
            min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]
        ]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_youtube(
        self, labels, ctx_l, max_n=1, add_easy_negative=False
    ):
        # Youtube-hl only have binary score
        agg_scores = np.array(labels)[:, 0]  # (L, 1) --> (L, )
        score_array = agg_scores * 1

        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[: self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (
                ctx_l * self.clip_len
            )  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor(
                [
                    [
                        int(w[0] / self.clip_len),
                        min(int(w[1] / self.clip_len), ctx_l) - 1,
                    ]
                    for w in windows
                ]
            ).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()]
        )
        return self.embedding(word_inds)

    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == "tvsum" or self.dset_name == "youtube_highlight":
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)
            return (
                torch.from_numpy(q_feat["token"])
                if self.dset_name == "tvsum"
                else torch.from_numpy(q_feat["last_hidden_state"])
            )

        else:
            if self.dset_name == "tacos":
                q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            elif "subs_train" in self.data_path:  # for pretrain
                vid = "_".join(qid.split("_")[:-1])
                subid = qid.split("_")[-1]
                q_feat_path = join(self.q_feat_dir, f"{vid}/{subid}.npz")
            else:
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")

            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[: self.max_q_l]
            q_feat = l2_normalize_np_array(q_feat)
            return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            if self.dset_name == "tvsum" and "i3d" in _feat_dir:
                rgb_path = join(_feat_dir, f"{vid}_rgb.npy")
                opt_path = join(_feat_dir, f"{vid}_opt.npy")
                rgb_feat = np.load(rgb_path)[: self.max_v_l].astype(np.float32)
                opt_feat = np.load(opt_path)[: self.max_v_l].astype(np.float32)
                _feat = np.concatenate([rgb_feat, opt_feat], axis=-1)
                _feat = l2_normalize_np_array(_feat)  # normalize?
                v_feat_list.append(_feat)
            else:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"][: self.max_v_l].astype(
                    np.float32
                )
                _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)

        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

    def _get_audio_feat_by_vid(self, vid):
        a_feat_list = []
        for _feat_dir in self.a_feat_dirs:
            if self.a_feat_types == "clap":
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"][: self.max_a_l].astype(
                    np.float32
                )
            else:
                raise NotImplementedError
            _feat = l2_normalize_np_array(_feat)  # normalize?
            a_feat_list.append(_feat)

        # some features are slightly longer than the others
        min_len = min([len(e) for e in a_feat_list])
        a_feat_list = [e[:min_len] for e in a_feat_list]
        a_feat = np.concatenate(a_feat_list, axis=1)
        return torch.from_numpy(a_feat)  # (Lv, D)


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [
                dict(spans=e["model_inputs"]["span_labels"]) for e in batch
            ]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch],
                dtype=np.float32,
                fixed_length=None,
            )
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch],
            dtype=torch.float32,
            fixed_length=None,
        )
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(
            device, non_blocking=non_blocking
        ),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(
            device, non_blocking=non_blocking
        ),
        src_vid=batched_model_inputs["video_feat"][0].to(
            device, non_blocking=non_blocking
        ),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(
            device, non_blocking=non_blocking
        ),
    )

    if "audio_feat" in batched_model_inputs:
        model_inputs["src_aud"] = batched_model_inputs["audio_feat"][0].to(
            device, non_blocking=non_blocking
        )
        model_inputs["src_aud_mask"] = batched_model_inputs["audio_feat"][1].to(
            device, non_blocking=non_blocking
        )

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(
                device, non_blocking=non_blocking
            )

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(
            device, non_blocking=non_blocking
        )

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets


