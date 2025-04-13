from time import perf_counter

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverStatus, TerminationCondition

from LiCE.data.Types import DataLike
from LiCE.spn.SPN import SPN

from .spn_enc import encode_spn


class LiCE:
    MIO_EPS = 1e-6

    def __init__(
        self,
        B: np.ndarray,
        max_stars: int = 1,  # for 0/1 case
        spn: SPN | None = None,
        groups: (
            list[list[str]] | None
        ) = None,  # list of groups defined by list of items in them
        grouping_f: str = "none",  # sum, mean, disjunction or none
    ):
        self.__B = B.astype(float)
        self.__M = (
            B[B > 0].sum(axis=0).max() - B[B < 0].sum(axis=0).min()
        )  # could be computed only for the spots where the facual has ones
        self.__max_stars = max_stars
        if spn is not None:
            self.__spn = spn
            self.__groups = groups
            # if groups is not None:
            #     self.__groups = [[items.index(i) for i in g] for g in groups]
            self.__grouping_f = grouping_f

    def __build_model(
        self,
        factual: np.ndarray,
        weights: np.ndarray,
        ll_threshold: float = -np.inf,
        optimize_ll: bool = False,
        # prediction_threshold: float = 1e-4,
        ll_opt_coef: float = 0.1,
        leaf_encoding: float = "histogram",
        spn_variant: str = "lower",
        selected_items: list[int] | None = None,
    ) -> pyo.Model:
        model = pyo.ConcreteModel()

        model.in_set = pyo.Set(initialize=[i for i, f in enumerate(factual) if f != 0])
        if self.__max_stars == 1:
            model.change = pyo.Var(model.in_set, domain=pyo.Binary)
        else:
            model.change = pyo.Var(
                model.in_set,
                domain=pyo.Integers,
                bounds=lambda m, i: (0, round(factual[i] * self.__max_stars)),
            )

        if selected_items is not None:
            model.out_set = pyo.Set(initialize=selected_items)
        else:
            model.out_set = pyo.Set(initialize=list(range(factual.shape[0])))

        if optimize_ll or ll_threshold > -np.inf:
            if self.__grouping_f == "none":
                model.spn_i = pyo.Set(initialize=np.arange(factual.shape[0]))
                model.spn_in = pyo.Var(model.spn_i, domain=model.change.domain)
                model.spnInConstrNone = pyo.Constraint(
                    model.spn_i,
                    rule=lambda m, i: (
                        round(factual[i] * self.__max_stars) - m.change[i]
                        == m.spn_in[i]
                    ),
                )
            else:
                model.spn_i = pyo.Set(initialize=list(range(len(self.__groups))))

                if self.__grouping_f == "disjunction":
                    model.spn_in = pyo.Var(model.spn_i, domain=pyo.Binary)
                    model.spnInConstr0 = pyo.Constraint(
                        model.spn_i,
                        rule=lambda m, i: (
                            sum(
                                factual[j] - m.change[j] / self.__max_stars
                                for j in self.__groups[i]
                                if j in m.in_set
                            )
                            >= m.spn_in[i] / self.__max_stars
                        ),
                    )
                    model.spnInConstr1 = pyo.Constraint(
                        model.spn_i,
                        rule=lambda m, i: (
                            sum(
                                factual[j] - m.change[j] / self.__max_stars
                                for j in self.__groups[i]
                                if j in m.in_set
                            )
                            <= m.spn_in[i] * len(self.__groups[i])
                        ),
                    )
                elif self.__grouping_f == "sum":
                    model.spn_in = pyo.Var(
                        model.spn_i,
                        domain=pyo.Reals,
                        bounds=lambda m, i: (0, len(self.__groups[i])),
                    )
                    model.spnInConstrSum = pyo.Constraint(
                        model.spn_i,
                        rule=lambda m, i: (
                            sum(
                                factual[j] - m.change[j] / self.__max_stars
                                for j in self.__groups[i]
                                if j in m.in_set
                            )
                            == m.spn_in[i]
                        ),
                    )
                elif self.__grouping_f == "mean":
                    model.spn_in = pyo.Var(model.spn_i, domain=pyo.Reals, bounds=(0, 1))
                    model.spnInConstrSum = pyo.Constraint(
                        model.spn_i,
                        rule=lambda m, i: (
                            sum(
                                factual[j] - m.change[j] / self.__max_stars
                                for j in self.__groups[i]
                                if j in m.in_set
                            )
                            / len(self.__groups[i])
                            == m.spn_in[i]
                        ),
                    )

            spn_inputs = model.spn_in

        if optimize_ll:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn,
                model.spn,
                spn_inputs,
                leaf_encoding=leaf_encoding,
                mio_epsilon=self.MIO_EPS,
                sum_approx=spn_variant,
            )
            # set up objective
            model.obj = pyo.Objective(
                expr=sum(
                    model.change[i] * weights[i] / self.__max_stars
                    for i in model.in_set
                )
                - ll_opt_coef * spn_outputs[self.__spn.out_node_id],
                sense=pyo.minimize,
            )
            return model

        elif ll_threshold > -np.inf:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn,
                model.spn,
                spn_inputs,
                leaf_encoding=leaf_encoding,
                mio_epsilon=self.MIO_EPS,
                sum_approx=spn_variant,
            )
            model.ll_constr = pyo.Constraint(
                expr=spn_outputs[self.__spn.out_node_id] >= ll_threshold
            )

        # set up objective
        model.obj = pyo.Objective(
            expr=sum(
                model.change[i] * weights[i] / self.__max_stars for i in model.in_set
            ),
            sense=pyo.minimize,
        )

        return model

    def __set_up_below_score(
        self, model, factual: np.ndarray, item_i: int, score: float
    ):
        model.below_score = pyo.Constraint(
            expr=sum(
                (factual[i] - model.change[i] / self.__max_stars) * self.__B[i, item_i]
                for i in model.in_set
            )
            <= score
        )

    def __set_up_ordering(self, model, factual: np.ndarray, ordering: list[int]):
        model.order_range = pyo.Set(initialize=range(len(ordering)))
        model.out = pyo.Var(model.order_range, bounds=(None, None), domain=pyo.Reals)
        model.out_constr = pyo.Constraint(
            model.order_range,
            rule=lambda m, ord_i: sum(
                (factual[i] - m.change[i] / self.__max_stars)
                * self.__B[i, ordering[ord_i]]
                for i in m.in_set
            )
            == m.out[ord_i],
        )
        model.order_constr = pyo.Constraint(
            model.order_range,
            rule=lambda m, ord_i: (
                m.out[ord_i] >= m.out[ord_i + 1]
                if ord_i + 1 in m.order_range
                else pyo.Constraint.Skip
            ),
        )

    def __set_up_nth(self, model, factual: np.ndarray, picked_i: int, nth_place: int):
        model.out = pyo.Var(model.out_set, bounds=(None, None), domain=pyo.Reals)
        model.out_constr = pyo.Constraint(
            model.out_set,
            rule=lambda m, out_i: (
                sum(
                    (factual[i] - m.change[i] / self.__max_stars) * self.__B[i, out_i]
                    for i in m.in_set
                )
                == m.out[out_i]
            ),
        )
        model.better = pyo.Var(model.out_set, domain=pyo.Binary)

        model.picks = pyo.Constraint(
            model.out_set,
            rule=lambda m, out_i: (
                m.out[picked_i] - m.out[out_i] <= (1 - m.better[out_i]) * self.__M
                if out_i != picked_i
                else pyo.Constraint.Skip
            ),
        )

        model.fix_picked = pyo.Constraint(expr=model.better[picked_i] == 0)
        model.placement = pyo.Constraint(
            expr=sum(b for b in model.better.values()) >= nth_place - 1
        )

    def generate_counterfactual(
        self,
        factual: np.ndarray,  # assumed in the correct item order...
        selected_item: int,  # assumed to be the correct index...
        score: float | None = None,
        nth: int | None = None,
        weights: np.ndarray | None = None,
        allow_self: bool = True,
        ll_threshold: float = -np.inf,
        ll_opt_coefficient: float = 0,
        n_counterfactuals: int = 1,
        solver_name: str = "appsi_highs",
        verbose: bool = False,
        time_limit: int = 600,
        leaf_encoding: str = "histogram",
        spn_variant: str = "lower",
        ce_relative_distance: float = np.inf,
        ce_max_distance: float = np.inf,
    ) -> tuple[bool, list[np.ndarray]]:

        if weights is None:
            weights = np.ones_like(factual)

        t_start = perf_counter()
        model = self.__build_model(
            factual,
            weights,
            ll_threshold,
            ll_opt_coefficient != 0,
            leaf_encoding=leaf_encoding,
            ll_opt_coef=ll_opt_coefficient,
            spn_variant=spn_variant,
        )
        if not allow_self:
            model.block_self = pyo.Constraint(expr=model.change[selected_item] == 0)

        if nth is not None:
            # it must be at a given position
            self.__set_up_nth(model, factual, selected_item, nth)
        elif score is not None:
            # we reduce its score to a fixed value
            self.__set_up_below_score(model, factual, selected_item, score)
        t_built = perf_counter()

        if solver_name == "gurobi":
            opt = pyo.SolverFactory(solver_name, solver_io="python")
        else:
            opt = pyo.SolverFactory(solver_name)

        if n_counterfactuals > 1:
            if solver_name != "gurobi":
                raise NotImplementedError(
                    "Generating multiple counterfactuals is supported only for Gurobi solver"
                )
            opt.options["PoolSolutions"] = n_counterfactuals  # Store n solutions
            opt.options["PoolSearchMode"] = 2  # Systematic search for n-best solutions
            if ce_relative_distance != np.inf:
                # Accept solutions within ce_relative_distance*100% of the optimal
                opt.options["PoolGap"] = ce_relative_distance
        if ce_max_distance != np.inf:
            print("Limiting max distance by", ce_max_distance)
            model.max_dist = pyo.Constraint(
                expr=model.input_encoding.total_cost <= ce_max_distance
            )

        if "cplex" in solver_name:
            opt.options["timelimit"] = time_limit
        elif "glpk" in solver_name:
            opt.options["tmlim"] = time_limit
        elif "xpress" in solver_name:
            opt.options["soltimelimit"] = time_limit
            # Use the below instead for XPRESS versions before 9.0
            # self.solver.options['maxtime'] = TIME_LIMIT
        elif "highs" in solver_name:
            opt.options["time_limit"] = time_limit
        elif solver_name == "gurobi":
            opt.options["TimeLimit"] = time_limit
            # opt.options["Aggregate"] = 0
            # opt.options["OptimalityTol"] = 1e-3
            opt.options["IntFeasTol"] = self.MIO_EPS / 10
            opt.options["FeasibilityTol"] = self.MIO_EPS / 10
        else:
            print("Time limit not set! Not implemented for your solver")

        t_prepped = perf_counter()
        result = opt.solve(model, load_solutions=False, tee=verbose)
        t_solved = perf_counter()

        self.__t_build = t_built - t_start
        self.__t_solve = t_solved - t_prepped
        self.__model = model
        self.__loglikelihoods = []
        self.__distances = []

        if verbose:
            if "gurobi" in solver_name:
                opt._solver_model.printStats()
            print(result)
        if result.solver.status == SolverStatus.ok:
            if result.solver.termination_condition == TerminationCondition.optimal:
                # print(pyo.value(model.obj))
                # print(model.spn.node_out[self.__spn.out_node_id].value)
                model.solutions.load_from(result)
                CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
                self.__t_tot = perf_counter() - t_start
                self.__optimal = True
                return CEs
        elif result.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
            # the objective value is always bounded
        ]:
            print("Infeasible formulation")
            if verbose:
                write_iis(model, "IIS.ilp", solver="gurobi")
            self.__t_tot = (perf_counter() - t_start,)
            self.__optimal = False
            return []
        elif (
            result.solver.status == SolverStatus.aborted
            and result.solver.termination_condition == TerminationCondition.maxTimeLimit
        ):
            print("TIME LIMIT")
            try:
                model.solutions.load_from(result)
            except ValueError:
                self.__t_tot = (perf_counter() - t_start,)
                self.__optimal = False
                return []
            CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
            self.__t_tot = (perf_counter() - t_start,)
            self.__optimal = False
            return CEs
        # else:

        self.__t_tot = (perf_counter() - t_start,)
        self.__optimal = False
        # print result if it wasn't printed yet
        if not verbose:
            print(result)
        raise ValueError("Unexpected termination condition")

    def __get_CEs(
        self, n: int, model: pyo.Model, factual: np.ndarray, opt: pyo.SolverFactory
    ):
        self.__loglikelihoods = []
        self.__distances = []
        if n > 1:
            raise ValueError("this is not re implemented for recommenders")
        else:
            self.__distances.append(
                sum(
                    self.__model.change[i].value / self.__max_stars
                    for i in self.__model.in_set
                )
            )
            if hasattr(self.__model, "spn"):
                self.__loglikelihoods.append(
                    self.__model.spn.node_out[self.__spn.out_node_id].value
                )
            res = factual
            for i in self.__model.in_set:
                res[i] -= self.__model.change[i].value / self.__max_stars
            return [res]

    @property
    def stats(self):
        return {
            "time_total": self.__t_tot,  # with CE recovery
            "time_solving": self.__t_solve,
            "time_building": self.__t_build,
            "optimal": self.__optimal,
            "ll_computed": self.__loglikelihoods,
            "dist_computed": self.__distances,
        }

    @property
    def model(self) -> pyo.Model:
        return self.__model
