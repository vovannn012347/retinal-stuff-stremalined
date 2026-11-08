"""SurgicalSlicing schedule (pure logic, no torch) – one decision per pass.

cycle:  LEARN  -> gate reached ------------------------------> accept (save) -> RAMP
        RAMP   -> dropout += step every pass up to dropout_max (only on configured layers)
                  dropout >= penalty_start: select units, penalty += step*base every pass
                  dropout == max and penalty == max (+ hold_passes_at_max) -> CUT
        CUT    -> (done by the trainer) -> LEARN of the next cycle
        LEARN fails to reach the gate in time -> FAIL (trainer restores the last accepted
                  model and retries with a smaller cut, or stops)
"""

EPS = 1e-9


class SurgicalSchedule:
    def __init__(self, cfg):
        self.c = cfg
        self.phase = 'learn'
        self.cycle = 0
        self.phase_passes = 0
        self.dropout = float(cfg['learn_dropout'])
        self.penalty = 0.0
        self.penalty_active = False
        self.holding = False
        self.hold_used = 0
        self.at_max_passes = 0
        self.cut_ratio = float(cfg['cut_ratio'])
        self.retries = 0
        self.gate_streak = 0

    # -- state ------------------------------------------------------------------------
    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'c'}

    def load_state_dict(self, d):
        self.__dict__.update(d)

    @property
    def penalty_max(self):
        return float(self.c['penalty_base'])

    def max_learn(self):
        return int(self.c['max_learn_passes'] if self.cycle == 0 and self.retries == 0
                   else self.c['max_recover_passes'])

    # -- per pass ---------------------------------------------------------------------
    def begin_pass(self):
        """Returns dict(dropout, penalty, select_now, apply_penalty)."""
        select_now = False
        if self.phase == 'learn':
            self.dropout = float(self.c['learn_dropout'])
        elif self.phase == 'ramp' and not self.holding:
            self.dropout = min(float(self.c['dropout_max']), self.dropout + float(self.c['dropout_step']))
            self.dropout = round(self.dropout, 6)
            if self.dropout + EPS >= float(self.c['penalty_start_dropout']):
                if not self.penalty_active:
                    self.penalty_active, select_now, self.penalty = True, True, 0.0
                step = float(self.c['penalty_step']) * self.penalty_max
                self.penalty = round(min(self.penalty_max, self.penalty + step), 6)
                # snap to the max when the remainder is just rounding (0.066 * 15 = 0.99)
                if self.penalty_max - self.penalty < 0.5 * step:
                    self.penalty = self.penalty_max
        apply_penalty = self.phase == 'ramp' and self.penalty_active and not self.holding
        return {'dropout': self.dropout, 'penalty': self.penalty if self.penalty_active else 0.0,
                'select_now': select_now, 'apply_penalty': apply_penalty, 'holding': self.holding}

    def end_pass(self, gate_passed: bool, degraded: bool):
        """Returns one of: continue | accept | cut | fail."""
        self.phase_passes += 1
        if self.phase == 'learn':
            self.gate_streak = self.gate_streak + 1 if gate_passed else 0
            if (self.gate_streak >= int(self.c.get('gate_consecutive_passes', 1))
                    and self.phase_passes >= int(self.c['min_learn_passes'])):
                return 'accept'
            if self.phase_passes >= self.max_learn():
                return 'fail'
            return 'continue'

        # ramp
        at_max = (self.dropout + EPS >= float(self.c['dropout_max'])
                  and self.penalty_active and self.penalty + EPS >= self.penalty_max)
        if at_max and not self.holding:
            if self.at_max_passes >= int(self.c['hold_passes_at_max']):
                return 'cut'
            self.at_max_passes += 1
        self.holding = bool(degraded and self.hold_used < int(self.c['max_hold_passes']))
        if self.holding:
            self.hold_used += 1
        return 'continue'

    # -- transitions (called by the trainer) ------------------------------------------
    def start_ramp(self):
        self.phase, self.phase_passes = 'ramp', 0
        self.dropout = float(self.c['learn_dropout'])
        self.penalty, self.penalty_active = 0.0, False
        self.holding, self.hold_used, self.at_max_passes = False, 0, 0

    def after_cut(self):
        self.cycle += 1
        self.phase, self.phase_passes, self.gate_streak = 'learn', 0, 0
        self.penalty, self.penalty_active = 0.0, False
        self.holding, self.hold_used, self.at_max_passes = False, 0, 0

    def after_accept(self):
        self.retries = 0

    def retry_smaller(self):
        """Recovery failed: cut less next time. Returns False when no retry is left."""
        self.retries += 1
        self.cycle = max(0, self.cycle - 1)          # the cycle is redone from the last accepted model
        self.cut_ratio *= float(self.c['retry_ratio_factor'])
        if self.retries > int(self.c['max_retries']) or self.cut_ratio < float(self.c['min_cut_ratio']):
            return False
        self.start_ramp()
        return True
