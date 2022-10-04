import cython

def check_continuous_collision(float from1x, float from1y, float to1x, float to1y, float rad1, 
            float from2x, float from2y, float to2x, float to2y, float rad2):
    
    def dist_pow2(float t):
        m1x = (1 - t) * from1x + t * to1x
        m1y = (1 - t) * from1y + t * to1y
        m2x = (1 - t) * from2x + t * to2x
        m2y = (1 - t) * from2y + t * to2y
        return (m1x - m2x) ** 2 + (m1y - m2y) ** 2
    
    corrx = -from1x + to1x + from2x - to2x
    corry = -from1y + to1y + from2y - to2y
    a = corrx ** 2 + corry ** 2
    b = corrx * (from1x - from2x) + corry * (from1y - from2y)

    if b >= 0:  # two agents are going away
        min_dist_pow2 = dist_pow2(0)
    elif a + b <= 0:  # two agents are approaching
        min_dist_pow2 = dist_pow2(1)
    else:  # otherwise
        min_dist_pow2 = dist_pow2(-b / a)

    return min_dist_pow2 <= (rad1 + rad2) ** 2