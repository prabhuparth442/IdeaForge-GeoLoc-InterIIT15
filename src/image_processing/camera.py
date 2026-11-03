def distort(camera, x, y):
    r2 = x * x + y * y
    radial = 1 + camera.k1 * r2 + camera.k2 * r2 ** 2 + camera.k3 * r2 ** 3
    x_d = x * radial + 2 * camera.p1 * x * y + camera.p2 * (r2 + 2 * x * x)
    y_d = y * radial + camera.p1 * (r2 + 2 * y * y) + 2 * camera.p2 * x * y
    return x_d, y_d
