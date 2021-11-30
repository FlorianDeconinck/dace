from os import name
from typing import Dict, List, Union
import sympy as sp
from dace.sdfg.graph import SubgraphView
from dace.transformation.estimator.soap.sdg import SDG
from dace.transformation.estimator.soap.utils import parse_params, SOAPParameters
from dace.transformation.estimator.soap.solver import Solver
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dataclasses import dataclass, field
from dace import SDFG
import networkx as nx

@dataclass
class IOAnalysisSubgraph():
    name : int
    Q: sp.Expr
    rho : sp.Expr
    input_arrays : Dict
    variables :  List[sp.Expr]
    inner_tile : List[sp.Expr]
    outer_tile : List[sp.Expr]

    # The following three fields define the parallel distribution.
    # they require user to specify numerical value of parameters, 
    # such as P, S, and problem sizes. The default values of some of these 
    # parameters are specified in utils.param_values
    loc_domain_dims : List[sp.Expr] = field(default_factory=list)
    p_grid : List[sp.Expr] = field(default_factory=list)
    dimensions_ordered : List[sp.Expr] = field(default_factory=list)

    def get_data_decomposition(self, pp):
        # p_pos is an N -> N^d mapping, translating a global rank to a cooridnate rank in 
        # the d-dimensional iteration space  (p  -> [p_x, p_y, p_z, ....])
        if (sp.prod(self.p_grid) <= pp):
            print("\n\nDecomposition uses only {} processors. Given rank {} will be idle!\n\n".format(sp.prod(self.p_grid), pp))
            return {}
        
        div = lambda x,y: (x-x%y)/y
        p_pos = []
        for dim_num, dim_size in enumerate(self.p_grid):
            p_dim = div(pp, sp.prod(self.p_grid[dim_num+1 :]))
            p_pos.append(p_dim)
            pp -= p_dim * sp.prod(self.p_grid[dim_num+1 :])
            
        iter_ranges = {str(var) : 
            (p_pos[i]* self.loc_domain_dims[i], min((p_pos[i] + 1)* self.loc_domain_dims[i], dim_size))  
            for i, (var, dim_size) in enumerate(zip(self.variables, self.dimensions_ordered))}
        
        
        par_distribution = {}
        for arr, accesses in self.input_arrays.items():
            for access, pars in accesses.items():
                rngs = {}
                for it in access.split('*'):
                    rngs[it] = iter_ranges[it]
            par_distribution[arr] = rngs
                            
        return par_distribution   

@dataclass
class IOAnalysis():
    name : int
    Q: sp.Expr
    sdg: SDG
    subgraphs : list[IOAnalysisSubgraph]


def perform_soap_analysis(sdfg : Union[SDFG, SubgraphView],
                    generate_schedule : bool = False) -> IOAnalysis:
    """
    Main interface of the SOAP analysis. 

    Input:
    sdfg (SDFG): sdfg to be analyzed
    params [Optional] (global_parameters): User-defined SOAP parameters. Default values are defined in utils
    generate_schedule [Optional] (bool): Whether the parallel decomposition should be evaluated for each subgraph.
                                         The decomposition parameters are specified in params.param_values.

    Output:
    io_result_subgraph: dataclass containing SOAP analysis results, 
    such as I/O lower bound (Q), computational intensity (rho), symbolic directed graph (SDG),
    and the list of subgraphs (subgraphs) with their optimal tilings and kernel merging
    """
    if not params:
        params = parse_params()
        solver = Solver(cached_only = params.only_cached, caching_solutions= params.caching_solver_solutions)
        solver.start_solver(params.remoteMatlab)
        solver.set_timeout(300)
        params.solver = solver
        
    sdg = SDG(sdfg, params)
    # check if the created SDG is correct
    assert(nx.number_weakly_connected_components(sdg.graph) == 1)
    Q, subgraphs = sdg.calculate_IO_of_SDG(params)
    params.solver.end_solver()

    subgraphs_res = []
    for subgr in subgraphs:
        io_res_sg = IOAnalysisSubgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                    variables = subgr.variables, inner_tile = subgr.inner_tile, 
                    outer_tile = subgr.outer_tile, input_arrays = subgr.phis)                    
        if generate_schedule:
            subgr.init_decomposition(params.decomposition_params,  params)   
            io_res_sg.loc_domain_dims = subgr.loc_domain_dims
            io_res_sg.p_grid = subgr.p_grid
            io_res_sg.dimensions_ordered = subgr.dimensions_ordered        
        subgraphs_res.append(io_res_sg)
    
    return(IOAnalysis(SDFG.__name__, Q, sdg, subgraphs_res))



def perform_soap_analysis_einsum(einsum_string : str, params: SOAPParameters = [], 
            generate_schedule : bool = True) -> IOAnalysis:
    """
    Specialization of the main interface dedicated for the einsum tensor operations.

    Input:
    einsum_str: einsum srtring to be analyzed

    Output:
    io_result_subgraph: dataclass containing SOAP analysis results, 
    such as I/O lower bound (Q), computational intensity (rho), symbolic directed graph (SDG),
    and the list of subgraphs (subgraphs) with their optimal tilings and kernel merging
    """
    if params == []:
        params = parse_params()
        solver = Solver(cached_only = params.only_cached, caching_solutions= params.caching_solver_solutions)
        solver.start_solver(params.remoteMatlab)
        solver.set_timeout(300)
        params.solver = solver

    sdfg = sdfg_gen(einsum_string)
    sdg = SDG(sdfg, params)
    Q, subgraphs= sdg.calculate_IO_of_SDG(params)

    subgraphs_res = []
    for subgr in subgraphs:
        io_res_sg = IOAnalysisSubgraph(name= subgr.name, Q = subgr.Q, rho = subgr.rhoOpts, 
                    variables = subgr.variables, inner_tile = subgr.inner_tile, 
                    outer_tile = subgr.outer_tile, input_arrays = subgr.phis)                    
        if generate_schedule:
            subgr.init_decomposition(params.decomposition_params,  params)
        
        io_res_sg.loc_domain_dims = subgr.loc_domain_dims
        io_res_sg.p_grid = subgr.p_grid
        io_res_sg.dimensions_ordered = subgr.dimensions_ordered        
        subgraphs_res.append(io_res_sg)
    
    return(IOAnalysis(SDFG.__name__, Q, sdg, subgraphs_res))

