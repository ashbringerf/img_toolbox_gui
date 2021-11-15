import matplotlib.pyplot as plt
import cv2
import numpy as np
from color_ulti import tsplit, tstack, Lab2LCh, LCh2Lab, cal_deltaAB, deltaE00


def cal_deltaE00_from_LCh(LCh_1, Lab_2):
    """
    Calculate the color difference :math:`\Delta E_{00}` between two given colorspace arrays.

    :param LCh_1: array-like
    :param Lab_2: array-like
    :return: numeric or ndarray
    """
    Lab_1 = LCh2Lab(LCh_1)
    return deltaE00(Lab_1, Lab_2)


def cal_deltaE00_residual(C, target, Lab, L, h):
    """
    Calculate the residual between the target and the :math:`\Delta E_{00}` value in given LCh and Lab.

    :param C: numeric
    :param target: numeric
    :param Lab: array-like
    :param L: numeric
    :param h: numeric
    :return: numeric
    """
    local_var_LCh = tstack([L, C, h])
    residual = cal_deltaE00_from_LCh(local_var_LCh, Lab) - target
    return residual


def one_dimensional_search(x_start, x_end, Lab_reference, target, epsilon=0.01):
    """
    Return the optimal solution :math:`C^{*}` which minimize the residual between target and :math:`\Delta E_{00}`.

    :param x_start: numeric
    :param x_end: numeric
    :param Lab_reference: array-like
    :param target: numeric
    :param epsilon: numeric
    :return: numeric
    """
    [L0, _, h0] = tsplit(Lab2LCh(Lab_reference))
    x1 = x_start + 0.618 * (x_end - x_start)
    x2 = x_start + x_end - x1
    y1 = -np.fabs(cal_deltaE00_residual(x1, target, Lab_reference, L0, h0))
    y2 = -np.fabs(cal_deltaE00_residual(x2, target, Lab_reference, L0, h0))
    if y1 > y2:
        if np.fabs(y1) < epsilon:
            return x1
        else:
            return one_dimensional_search(x2, x_end, Lab_reference, target, epsilon)
    elif y1 < y2:
        if np.fabs(y2) < epsilon:
            return x2
        else:
            return one_dimensional_search(x_start, x1, Lab_reference, target, epsilon)
    elif y1 == y2:
        if np.fabs(y2) < epsilon:
            return x2
        else:
            return one_dimensional_search(x2, x1, Lab_reference, target, epsilon)


def chroma_shift(Lab_reference, chroma, epsilon=0.01):
    """
    Return a Lab array which has given chroma(:math:`\Delta E_{00}`) with Lab_reference.

    :param Lab_reference: array-like
    :param chroma: numeric
    :param epsilon: numeric
    :return: ndarray
    """
    Lab_reference = np.squeeze(Lab_reference)
    [L, C, h] = tsplit(Lab2LCh(Lab_reference))
    target = np.fabs(chroma)
    if chroma > 0 and cal_deltaE00_from_LCh(tstack([L, 176, h]), Lab_reference) < target:
        return Lch2Lab(tstack([L, 125, h]))
    elif chroma < 0 and cal_deltaE00_from_LCh(tstack([L, 0, h]), Lab_reference) < target:
        return tstack([L, 0, 0])
    elif chroma == 0:
        return Lab_reference

    local_var_deltaE = target + 1
    root = 1
    C1 = C
    while local_var_deltaE > target:
        C1 = C + (0.1 ** root) * chroma
        local_var_deltaE = cal_deltaE00_from_LCh(tstack([L, C1, h]), Lab_reference)
        root += 1
    local_var_deltaE = target - 1
    root = 0
    C3 = C
    while local_var_deltaE < target:
        C3 = C + (3 ** root) * chroma
        local_var_deltaE = cal_deltaE00_from_LCh(tstack([L, C3, h]), Lab_reference)
        root += 1
    C_star = C
    if chroma > 0:
        C_star = one_dimensional_search(np.max([C1, 0]), np.min([C3, 176]), Lab_reference, target, epsilon)
    elif chroma < 0:
        C_star = one_dimensional_search(np.max([C3, 0]), np.min([C1, 176]), Lab_reference, target, epsilon)

    # r = cal_deltaE00_residual(C_star,target,Lab,L,h)
    # print(r)
    return LCh2Lab(tstack([L, C_star, h]))


def get_ellipse(e_x, e_y, a, b, e_angle):
    """
    Return the ellipse trace based given ellipse center (e_x,e_y), half long-axis length a, half short-axis length b,
    rotation angle e_angle(degree).

    :param e_x: numeric
    :param e_y: numeric
    :param a: numeric
    :param b: numeric
    :param e_angle: numeric
    :return: ndarray
    """
    angles_circle = np.arange(0, 2 * np.pi, 0.001)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * np.cos(angles)
        or_y = b * np.sin(angles)
        length_or = np.sqrt(or_x * or_x + or_y * or_y)
        or_theta = np.arctan2(or_y, or_x)
        new_theta = or_theta + np.deg2rad(e_angle)
        new_x = e_x + length_or * np.cos(new_theta)
        new_y = e_y + length_or * np.sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y


def fit_elli_trace(Lab_reference, JND, epsilon=0.01):
    """
    Return the ellipse trace based given Lab_reference and JND.

    :param Lab_reference: array-like
    :param JND: numeric
    :param epsilon: numeric
    :return: ndarray
    """
    Lab_reference = np.squeeze(Lab_reference)
    local_var_Lab1 = chroma_shift(Lab_reference, JND, epsilon)
    local_var_Lab2 = chroma_shift(Lab_reference, -JND, epsilon)
    local_var_Lab3 = find_pts_on_orth(Lab_reference, JND, epsilon)
    local_var_Lab4 = find_pts_on_orth(Lab_reference, -JND, epsilon)
    [_, a0, b0] = tsplit(Lab_reference)
    [_, _, h] = tsplit(Lab2LCh(Lab_reference))
    [_, C1, _] = tsplit(Lab2LCh(local_var_Lab1))
    [_, C2, _] = tsplit(Lab2LCh(local_var_Lab2))
    l1 = 0.5 * np.fabs(C1 - C2)
    l2 = 0.5 * cal_deltaAB(local_var_Lab3,local_var_Lab4)
    x, y = get_ellipse(a0, b0, l1, l2, h)
    return x, y


def cal_deltaE00_residual_4_orth(Lab_1, Lab_2, target):
    """
    Calculate the residual between target and :math:`\Delta E_{00}`.

    :param Lab_1: array-like
    :param Lab_2: array-like
    :param target: numeric
    :return: numeric
    """
    residual = deltaE00(Lab_1, Lab_2) - target
    return residual


def one_dimensional_search_4_orth(x_start, x_end, param, target, epsilon=0.01):
    """
    Return the optimal solution :math:`a^{*}` which minimize the residual between target and :math:`\Delta E_{00}`.

    :param x_start: numeric
    :param x_end: numeric
    :param param: [numeric, numeric, array-like]
    :param target: numeric
    :param epsilon: numeric
    :return: numeric
    """
    [k, b, Lab_reference] = param
    [L0, _, _] = tsplit(Lab_reference)
    a1 = x_start + 0.618 * (x_end - x_start)
    a2 = x_start + x_end - a1
    b1 = k * a1 + b
    b2 = k * a2 + b
    y1 = -np.fabs(cal_deltaE00_residual_4_orth(tstack([L0, a1, b1]), Lab_reference, target))
    y2 = -np.fabs(cal_deltaE00_residual_4_orth(tstack([L0, a2, b2]), Lab_reference, target))
    if y1 > y2:
        if np.fabs(y1) < epsilon:
            return a1
        else:
            return one_dimensional_search_4_orth(a2, x_end, (k, b, Lab_reference), target, epsilon)
    elif y1 < y2:
        if np.fabs(y2) < epsilon:
            return a2
        else:
            return one_dimensional_search_4_orth(x_start, a1, (k, b, Lab_reference), target, epsilon)
    elif y1 == y2:
        if np.fabs(y2) < epsilon:
            return a2
        else:
            return one_dimensional_search_4_orth(a2, a1, (k, b, Lab_reference), target, epsilon)


def one_dimensional_search_4_deg0(x_start, x_end, Lab_reference, target, epsilon=0.01):
    """
    Return the optimal solution :math:`b^{*}` which minimize the residual between target and :math:`\Delta E_{00}`.

    :param x_start: numeric
    :param x_end: numeric
    :param Lab_reference: array-like
    :param target: numeric
    :param epsilon: numeric
    :return: numeric
    """
    [L0, a0, _] = tsplit(Lab_reference)
    b1 = x_start + 0.618 * (x_end - x_start)
    b2 = x_start + x_end - b1
    a1 = a0
    a2 = a0
    y1 = -np.fabs(cal_deltaE00_residual_4_orth(tstack([L0, a1, b1]), Lab_reference, target))
    y2 = -np.fabs(cal_deltaE00_residual_4_orth(tstack([L0, a2, b2]), Lab_reference, target))
    if y1 > y2:
        if np.fabs(y1) < epsilon:
            return b1
        else:
            return one_dimensional_search_4_deg0(b2, x_end, Lab_reference, target, epsilon)
    elif y1 < y2:
        if np.fabs(y2) < epsilon:
            return b2
        else:
            return one_dimensional_search_4_deg0(x_start, b1, Lab_reference, target, epsilon)
    elif y1 == y2:
        if np.fabs(y2) < epsilon:
            return b2
        else:
            return one_dimensional_search_4_deg0(b2, b1, Lab_reference, target, epsilon)


def find_pts_on_orth(Lab_reference, chroma, epsilon=0.01):
    """
    Return Lab_star array which has given chroma(:math:`\Delta E_{00}`) with Lab_reference.

    :param Lab_reference: array-like
    :param chroma: numeric
    :param epsilon: numeric
    :return: ndarray
    """
    Lab_reference = np.squeeze(Lab_reference)
    target = np.fabs(chroma)
    [_, _, h0] = tsplit(Lab2LCh(Lab_reference))
    [L0, a0, b0] = tsplit(Lab_reference)
    a_star = a0
    b_star = b0
    if chroma == 0:
        return Lab_reference
    if h0 == 0 or h0 == 180:
        if chroma > 0:
            b_star = one_dimensional_search_4_deg0(b0, 125, Lab_reference, target, epsilon)
        elif chroma <= 0:
            b_star = one_dimensional_search_4_deg0(-125, b0, Lab_reference, target, epsilon)
        Lab_star = tstack([L0, a0, b_star])
        return Lab_star
    k = np.tan(np.deg2rad(h0 + 90))
    b = b0 - k * a0
    local_var_deltaE = target + 1
    root = 1
    a1 = a0
    while local_var_deltaE > target:
        a1 = a0 + (0.1 ** root) * chroma
        b1 = k * a1 + b
        local_var_deltaE = deltaE00(Lab_reference, tstack([L0, a1, b1]))
        root += 1
    local_var_deltaE = target - 1
    root = 0
    a2 = a0
    while local_var_deltaE < target and a2 < 125 and a2 > -125:
        a2 = a0 + (2 ** root) * chroma
        b2 = k * a2 + b
        local_var_deltaE = deltaE00(Lab_reference, tstack([L0, a2, b2]))
        root += 1
    if chroma > 0:
        a_star = one_dimensional_search_4_orth(np.max([-125, a1]), np.min([125, a2]),
                                               (k, b, Lab_reference), target, epsilon)
        b_star = k * a_star + b
    elif chroma < 0:
        a_star = one_dimensional_search_4_orth(np.max([-125, a2]), np.min([125, a1]),
                                               (k, b, Lab_reference), target, epsilon)
        b_star = k * a_star + b
    Lab_star = tstack([L0, a_star, b_star])
    # r = cal_deltaE00_residual_4_orth(Lab_star,Lab,target)
    # print(r)
    return Lab_star


if __name__ == '__main__':
    # Lab = np.array([[[23,13,4]],[[21,4,14]]])
    # Lab = [[[23,13,4]],[[21,4,14]]]
    # Lab = [[60,0,1],[50,1,0],[20,-30,0],[40,0,-2]]
    # Lch = Lab2LCh(Lab)

    Lab = [60,0, 50]
    x, y = fit_elli_trace(Lab, 2)
    print(x)
    # NewLab = chroma_shift(Lab,15,0.01)
    # print(NewLab)
    # Lch1 = Jab_to_JCh(Lab)
    # Lch2 = Jab_to_JCh(NewLab)
    # print(Lch1)
    # print(Lch2)
    # Lab_out = Lch2Lab(Lch)
    # Lab_out1 = JCh_to_Jab(Lch)
    # print(Lch)
    # [L,a,b] = tsplit(Lab)
    # print(L)
