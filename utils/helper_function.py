import multiprocessing
from utils.visualize_tools import plot_f_on_domain, plot_line


def petsc2array(v):
    s = v.getValues(range(0, v.getSize()[0]), range(0, v.getSize()[1]))
    return s


def visualize_result_dict(result_dict):
    marker_result = result_dict['marker_result']
    marker_exact = result_dict['marker_exact']

    p1 = multiprocessing.Process(
        target=plot_f_on_domain,
        kwargs={
            'domain': marker_result.function_space.mesh,
            'f': marker_result,
            'title': 'Ischemia Result',
        },
    )

    p2 = multiprocessing.Process(
        target=plot_f_on_domain,
        kwargs={
            'domain': marker_exact.function_space.mesh,
            'f': marker_exact,
            'title': 'Ischemia Exact',
        },
    )

    p3 = multiprocessing.Process(
        target=plot_line,
        kwargs={
            'line': result_dict['cm_cmp_per_iter'],
            'title': 'Center of Mass Error Over Iterations',
            'xlabel': 'Iteration',
            'ylabel': 'Center of Mass Error',
        },
    )

    p4 = multiprocessing.Process(
        target=plot_line,
        kwargs={
            'line': result_dict['loss_per_iter'],
            'title': 'Loss Over Iterations',
            'xlabel': 'Iteration',
            'ylabel': 'Loss',
        },
    )

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
