
def set_black_background(vis):
    opt = vis.get_render_option()
    opt.background_color = [0, 0, 0]  # 배경을 검은색으로 설정 (RGB값)
    return False

def set_white_background(vis):
    opt = vis.get_render_option()
    opt.background_color = [1, 1, 1]  # 배경을 검은색으로 설정 (RGB값)
    return False
