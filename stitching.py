'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """

    img_names = sorted(imgs.keys())
    if len(img_names) < 2:
        out_img = imgs[img_names[0]]
        if out_img.dtype == torch.uint8:
            return out_img
        return (out_img.clamp(0, 1) * 255).byte()

    img_a = imgs[img_names[0]]
    img_b = imgs[img_names[1]]

    if img_a.dtype == torch.uint8:
        img_a = img_a.float() / 255.0
    else:
        img_a = img_a.float()

    if img_b.dtype == torch.uint8:
        img_b = img_b.float() / 255.0
    else:
        img_b = img_b.float()

    device = img_a.device
    dtype = img_a.dtype

    def get_corners(image):
        gray = K.color.rgb_to_grayscale(image.unsqueeze(0))
        grad = K.filters.spatial_gradient(gray, mode='sobel', order=1)
        grad_x = grad[:, :, 0]
        grad_y = grad[:, :, 1]

        grad_x2 = K.filters.gaussian_blur2d(grad_x * grad_x, (5, 5), (1.0, 1.0))
        grad_y2 = K.filters.gaussian_blur2d(grad_y * grad_y, (5, 5), (1.0, 1.0))
        grad_xy = K.filters.gaussian_blur2d(grad_x * grad_y, (5, 5), (1.0, 1.0))

        harris = grad_x2 * grad_y2 - grad_xy * grad_xy - 0.04 * (grad_x2 + grad_y2) ** 2
        max_harris = torch.nn.functional.max_pool2d(harris, 7, 1, 3)

        mask = (harris == max_harris) & (harris > 1e-6)

        border = 11
        mask[:, :, :border, :] = False
        mask[:, :, -border:, :] = False
        mask[:, :, :, :border] = False
        mask[:, :, :, -border:] = False

        ys, xs = torch.where(mask[0, 0])
        if ys.numel() == 0:
            return torch.empty((0, 2), device=device, dtype=dtype), gray

        scores = harris[0, 0, ys, xs]
        keep_n = min(2500, scores.numel())
        _, top_ids = torch.topk(scores, keep_n)

        pts = torch.stack([xs[top_ids].float(), ys[top_ids].float()], dim=1)
        return pts, gray

    def get_patches(gray, pts):
        gray_img = gray[0, 0]
        patch_list = []

        for i in range(pts.shape[0]):
            x = int(round(float(pts[i, 0].item())))
            y = int(round(float(pts[i, 1].item())))

            patch = gray_img[y - 10:y + 11, x - 10:x + 11].reshape(-1)
            if patch.numel() != 441:
                continue

            patch = patch - patch.mean()
            patch = patch / (torch.norm(patch) + 1e-8)
            patch_list.append(patch)

        if len(patch_list) == 0:
            return torch.empty((0, 441), device=device, dtype=dtype)

        return torch.stack(patch_list)

    def match_points(desc_a, desc_b, ratio=0.85):
        if desc_a.shape[0] < 2 or desc_b.shape[0] < 2:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty

        dist = torch.cdist(desc_a, desc_b)

        best_vals, best_ids = torch.topk(dist, 2, largest=False)
        best_b = best_ids[:, 0]

        ratio_ok = best_vals[:, 0] < ratio * (best_vals[:, 1] + 1e-8)
        reverse_best = torch.argmin(dist, dim=0)

        a_ids = torch.arange(desc_a.shape[0], device=device)
        mutual_ok = reverse_best[best_b] == a_ids

        keep = ratio_ok & mutual_ok
        return a_ids[keep], best_b[keep]

    def find_homography(src_pts, dst_pts):
        if src_pts.shape[0] < 4:
            return None

        center_a = src_pts.mean(0)
        scale_a = (2 ** 0.5) / (torch.sqrt(((src_pts - center_a) ** 2).sum(1)).mean() + 1e-8)

        norm_a = torch.eye(3, device=device, dtype=dtype)
        norm_a[0, 0] = scale_a
        norm_a[1, 1] = scale_a
        norm_a[0, 2] = -scale_a * center_a[0]
        norm_a[1, 2] = -scale_a * center_a[1]

        center_b = dst_pts.mean(0)
        scale_b = (2 ** 0.5) / (torch.sqrt(((dst_pts - center_b) ** 2).sum(1)).mean() + 1e-8)

        norm_b = torch.eye(3, device=device, dtype=dtype)
        norm_b[0, 0] = scale_b
        norm_b[1, 1] = scale_b
        norm_b[0, 2] = -scale_b * center_b[0]
        norm_b[1, 2] = -scale_b * center_b[1]

        ones = torch.ones((src_pts.shape[0], 1), device=device, dtype=dtype)

        src_norm = (norm_a @ torch.cat([src_pts, ones], 1).t()).t()
        src_norm = src_norm[:, :2] / src_norm[:, 2:3]

        dst_norm = (norm_b @ torch.cat([dst_pts, ones], 1).t()).t()
        dst_norm = dst_norm[:, :2] / dst_norm[:, 2:3]

        x = src_norm[:, 0]
        y = src_norm[:, 1]
        u = dst_norm[:, 0]
        v = dst_norm[:, 1]

        n = src_pts.shape[0]
        A = torch.zeros((2 * n, 9), device=device, dtype=dtype)

        A[0::2, 0] = -x
        A[0::2, 1] = -y
        A[0::2, 2] = -1
        A[0::2, 6] = x * u
        A[0::2, 7] = y * u
        A[0::2, 8] = u

        A[1::2, 3] = -x
        A[1::2, 4] = -y
        A[1::2, 5] = -1
        A[1::2, 6] = x * v
        A[1::2, 7] = y * v
        A[1::2, 8] = v

        try:
            _, _, vh = torch.linalg.svd(A)
        except:
            return None

        H_norm = vh[-1].view(3, 3)
        H = torch.linalg.inv(norm_b) @ H_norm @ norm_a

        if torch.abs(H[2, 2]) < 1e-8:
            return None

        return H / H[2, 2]

    def run_ransac(src_pts, dst_pts):
        if src_pts.shape[0] < 4:
            return None, None

        best_H = None
        best_count = 0
        best_mask = None

        n = src_pts.shape[0]
        ones = torch.ones((n, 1), device=device, dtype=dtype)
        src_h = torch.cat([src_pts, ones], 1)

        for _ in range(8000):
            sample_ids = torch.randperm(n, device=device)[:4]
            H = find_homography(src_pts[sample_ids], dst_pts[sample_ids])

            if H is None:
                continue

            proj = (H @ src_h.t()).t()
            proj = proj[:, :2] / proj[:, 2:3]

            err = torch.sqrt(((proj - dst_pts) ** 2).sum(1))
            inlier_mask = err < 4

            count = int(inlier_mask.sum())

            if count > best_count:
                best_count = count
                best_H = H
                best_mask = inlier_mask

        if best_H is None or best_count < 4:
            return None, None

        better_H = find_homography(src_pts[best_mask], dst_pts[best_mask])
        if better_H is not None:
            best_H = better_H

        return best_H, best_mask

    pts_a, gray_a = get_corners(img_a)
    pts_b, gray_b = get_corners(img_b)

    if pts_a.shape[0] < 4 or pts_b.shape[0] < 4:
        c, h1, w1 = img_a.shape
        _, h2, w2 = img_b.shape

        out = torch.zeros((c, max(h1, h2), w1 + w2), device=device, dtype=dtype)
        out[:, :h1, :w1] = img_a
        out[:, :h2, w1:] = img_b

        return (out.clamp(0, 1) * 255).byte()

    desc_a = get_patches(gray_a, pts_a)
    desc_b = get_patches(gray_b, pts_b)

    H_ab = None

    for ratio in [0.8, 0.85, 0.9]:
        keep_a, keep_b = match_points(desc_a, desc_b, ratio)

        if keep_a.shape[0] < 4:
            continue

        match_a = pts_a[keep_a]
        match_b = pts_b[keep_b]

        H, inliers = run_ransac(match_a, match_b)

        if H is None:
            continue

        inlier_count = int(inliers.sum())

        if inlier_count < 6:
            continue

        H_ab = H
        break

    if H_ab is None:
        c, h1, w1 = img_a.shape
        _, h2, w2 = img_b.shape

        out = torch.zeros((c, max(h1, h2), w1 + w2), device=device, dtype=dtype)
        out[:, :h1, :w1] = img_a
        out[:, :h2, w1:] = img_b

        return (out.clamp(0, 1) * 255).byte()

    H_ba = torch.linalg.inv(H_ab)

    def get_warped_corners(H, w, h):
        corners = torch.tensor(
            [[0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]],
            device=device,
            dtype=dtype
        )
        ones = torch.ones((4, 1), device=device, dtype=dtype)

        corners_h = torch.cat([corners, ones], 1)
        warped = (H @ corners_h.t()).t()

        return warped[:, :2] / warped[:, 2:3]

    corners_a = get_warped_corners(torch.eye(3, device=device, dtype=dtype), img_a.shape[2], img_a.shape[1])
    corners_b = get_warped_corners(H_ba, img_b.shape[2], img_b.shape[1])

    all_corners = torch.cat([corners_a, corners_b])

    min_x = torch.floor(all_corners[:, 0].min())
    min_y = torch.floor(all_corners[:, 1].min())
    max_x = torch.ceil(all_corners[:, 0].max())
    max_y = torch.ceil(all_corners[:, 1].max())

    out_w = int((max_x - min_x + 1).item())
    out_h = int((max_y - min_y + 1).item())

    max_canvas = 8000

    if out_w <= 0 or out_h <= 0 or out_w > max_canvas or out_h > max_canvas or out_w * out_h > 20000000:
        print("Canvas too large -> skipping panorama:", out_w, out_h)
        out = img_list[anchor].clamp(0, 1) * 255.0
        return out.byte(), overlap.cpu()

    shift = torch.eye(3, device=device, dtype=dtype)
    shift[0, 2] = -float(min_x)
    shift[1, 2] = -float(min_y)

    warp_a_H = shift
    warp_b_H = shift @ H_ba

    warp_a = K.geometry.transform.warp_perspective(
        img_a.unsqueeze(0), warp_a_H.unsqueeze(0), (out_h, out_w),
        align_corners=True
    )

    warp_b = K.geometry.transform.warp_perspective(
        img_b.unsqueeze(0), warp_b_H.unsqueeze(0), (out_h, out_w),
        align_corners=True
    )

    mask_a = K.geometry.transform.warp_perspective(
        torch.ones((1, 1, img_a.shape[1], img_a.shape[2]), device=device, dtype=dtype),
        warp_a_H.unsqueeze(0), (out_h, out_w),
        align_corners=True
    ) > 0.5

    mask_b = K.geometry.transform.warp_perspective(
        torch.ones((1, 1, img_b.shape[1], img_b.shape[2]), device=device, dtype=dtype),
        warp_b_H.unsqueeze(0), (out_h, out_w),
        align_corners=True
    ) > 0.5

    mask_a = mask_a.float()
    mask_b = mask_b.float()

    only_a = mask_a * (1 - mask_b)
    only_b = mask_b * (1 - mask_a)
    both = mask_a * mask_b

    out = warp_a * only_a + warp_b * only_b

    diff = torch.mean(torch.abs(warp_a - warp_b), dim=1, keepdim=True)
    diff = K.filters.gaussian_blur2d(diff, (15, 15), (3, 3))

    avg_area = (diff < 0.06).float() * both

    take_a = (warp_a.mean(1, keepdim=True) >= warp_b.mean(1, keepdim=True)).float()
    take_a = K.filters.gaussian_blur2d(take_a, (21, 21), (5, 5))
    take_a = (take_a > 0.5).float() * both * (1 - avg_area)

    take_b = both * (1 - avg_area) * (1 - take_a)

    out = out + 0.5 * (warp_a + warp_b) * avg_area
    out = out + warp_a * take_a + warp_b * take_b

    out = out.squeeze(0).clamp(0, 1) * 255
    return out.byte()

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: NxN overlap matrix as torch.int64 tensor.
    """

    img_names = sorted(imgs.keys())
    n = len(img_names)

    if n == 0:
        return torch.zeros((3, 256, 256), dtype=torch.uint8), \
               torch.zeros((0, 0), dtype=torch.int64)

    img_list = []
    for name in img_names:
        cur_img = imgs[name]
        cur_img = cur_img.float() / 255.0 if cur_img.dtype == torch.uint8 else cur_img.float()
        img_list.append(cur_img)

    device = img_list[0].device
    dtype = img_list[0].dtype

    def get_points_and_gray(image):
        gray = K.color.rgb_to_grayscale(image.unsqueeze(0))
        grad = K.filters.spatial_gradient(gray, mode='sobel', order=1)

        grad_x = grad[:, :, 0]
        grad_y = grad[:, :, 1]

        grad_x2 = K.filters.gaussian_blur2d(grad_x * grad_x, (5, 5), (1.0, 1.0))
        grad_y2 = K.filters.gaussian_blur2d(grad_y * grad_y, (5, 5), (1.0, 1.0))
        grad_xy = K.filters.gaussian_blur2d(grad_x * grad_y, (5, 5), (1.0, 1.0))

        harris = grad_x2 * grad_y2 - grad_xy * grad_xy - 0.04 * (grad_x2 + grad_y2) ** 2
        pooled = torch.nn.functional.max_pool2d(harris, kernel_size=7, stride=1, padding=3)

        mask = (harris == pooled) & (harris > 1e-6)

        border = 11
        mask[:, :, :border, :] = False
        mask[:, :, -border:, :] = False
        mask[:, :, :, :border] = False
        mask[:, :, :, -border:] = False

        ys, xs = torch.where(mask[0, 0])

        if ys.numel() == 0:
            return torch.empty((0, 2), device=device, dtype=dtype), gray

        scores = harris[0, 0, ys, xs]
        keep_n = min(2500, scores.numel())
        _, top_ids = torch.topk(scores, k=keep_n, largest=True)

        pts = torch.stack([xs[top_ids].float(), ys[top_ids].float()], dim=1)
        return pts, gray

    def get_descriptors(gray, pts):
        gray_img = gray[0, 0]
        desc_list = []

        for i in range(pts.shape[0]):
            x = int(round(float(pts[i, 0].item())))
            y = int(round(float(pts[i, 1].item())))

            patch = gray_img[y - 10:y + 11, x - 10:x + 11].reshape(-1)
            if patch.numel() != 441:
                continue

            patch = patch - patch.mean()
            patch = patch / (torch.norm(patch) + 1e-8)
            desc_list.append(patch)

        if len(desc_list) == 0:
            return torch.empty((0, 441), device=device, dtype=dtype)

        return torch.stack(desc_list, dim=0)

    def match_features(desc_a, desc_b, ratio):
        if desc_a.shape[0] < 2 or desc_b.shape[0] < 2:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty

        dist = torch.cdist(desc_a, desc_b, p=2)
        vals, ids_ab = torch.topk(dist, k=2, dim=1, largest=False)

        best_ab = ids_ab[:, 0]
        ratio_ok = vals[:, 0] < ratio * (vals[:, 1] + 1e-8)

        best_ba = torch.argmin(dist, dim=0)
        a_ids = torch.arange(desc_a.shape[0], device=device)
        mutual_ok = best_ba[best_ab] == a_ids

        keep = ratio_ok & mutual_ok
        return a_ids[keep], best_ab[keep]

    def find_homography(src_pts, dst_pts):
        if src_pts.shape[0] < 4:
            return None

        center_a = src_pts.mean(0)
        scale_a = (2.0 ** 0.5) / (torch.sqrt(((src_pts - center_a) ** 2).sum(1) + 1e-8).mean() + 1e-8)

        norm_a = torch.eye(3, device=device, dtype=dtype)
        norm_a[0, 0] = scale_a
        norm_a[1, 1] = scale_a
        norm_a[0, 2] = -scale_a * center_a[0]
        norm_a[1, 2] = -scale_a * center_a[1]

        center_b = dst_pts.mean(0)
        scale_b = (2.0 ** 0.5) / (torch.sqrt(((dst_pts - center_b) ** 2).sum(1) + 1e-8).mean() + 1e-8)

        norm_b = torch.eye(3, device=device, dtype=dtype)
        norm_b[0, 0] = scale_b
        norm_b[1, 1] = scale_b
        norm_b[0, 2] = -scale_b * center_b[0]
        norm_b[1, 2] = -scale_b * center_b[1]

        ones = torch.ones((src_pts.shape[0], 1), device=device, dtype=dtype)

        src_norm = (norm_a @ torch.cat([src_pts, ones], 1).t()).t()
        src_norm = src_norm[:, :2] / (src_norm[:, 2:3] + 1e-8)

        dst_norm = (norm_b @ torch.cat([dst_pts, ones], 1).t()).t()
        dst_norm = dst_norm[:, :2] / (dst_norm[:, 2:3] + 1e-8)

        x = src_norm[:, 0]
        y = src_norm[:, 1]
        u = dst_norm[:, 0]
        v = dst_norm[:, 1]

        n_pts = src_pts.shape[0]
        A = torch.zeros((2 * n_pts, 9), device=device, dtype=dtype)

        A[0::2, 0] = -x
        A[0::2, 1] = -y
        A[0::2, 2] = -1
        A[0::2, 6] = x * u
        A[0::2, 7] = y * u
        A[0::2, 8] = u

        A[1::2, 3] = -x
        A[1::2, 4] = -y
        A[1::2, 5] = -1
        A[1::2, 6] = x * v
        A[1::2, 7] = y * v
        A[1::2, 8] = v

        try:
            _, _, vh = torch.linalg.svd(A)
        except RuntimeError:
            return None

        H_norm = vh[-1].view(3, 3)
        H = torch.linalg.inv(norm_b) @ H_norm @ norm_a

        if torch.abs(H[2, 2]) < 1e-8:
            return None

        return H / H[2, 2]

    def run_ransac(src_pts, dst_pts):
        if src_pts.shape[0] < 4:
            return None, None

        best_H = None
        best_inliers = None
        best_count = 0

        n_pts = src_pts.shape[0]
        ones = torch.ones((n_pts, 1), device=device, dtype=dtype)
        src_h = torch.cat([src_pts, ones], 1)

        for _ in range(8000):
            sample_ids = torch.randperm(n_pts, device=device)[:4]
            H = find_homography(src_pts[sample_ids], dst_pts[sample_ids])

            if H is None:
                continue

            proj = (H @ src_h.t()).t()
            proj = proj[:, :2] / (proj[:, 2:3] + 1e-8)

            err = torch.sqrt(((proj - dst_pts) ** 2).sum(1))
            inliers = err < 4.0
            count = int(inliers.sum().item())

            if count > best_count:
                best_count = count
                best_H = H
                best_inliers = inliers

        if best_H is None or best_count < 4:
            return None, None

        better_H = find_homography(src_pts[best_inliers], dst_pts[best_inliers])
        if better_H is not None:
            best_H = better_H

        return best_H, best_inliers

    all_points = []
    all_descs = []
    all_grays = []

    for image in img_list:
        pts, gray = get_points_and_gray(image)
        desc = get_descriptors(gray, pts)
        all_points.append(pts)
        all_descs.append(desc)
        all_grays.append(gray)

    overlap = torch.eye(n, device=device, dtype=torch.int64)
    pair_h = {}
    pair_score = {}

    for i in range(n):
        for j in range(i + 1, n):
            if all_points[i].shape[0] < 4 or all_points[j].shape[0] < 4:
                continue

            best_pair_h = None
            best_pair_count = 0

            for ratio in [0.80, 0.85, 0.90]:
                idx_a, idx_b = match_features(all_descs[i], all_descs[j], ratio=ratio)

                if idx_a.shape[0] < 4:
                    continue

                match_a = all_points[i][idx_a]
                match_b = all_points[j][idx_b]

                H, inliers = run_ransac(match_a, match_b)
                corn = torch.tensor([[0.,0.],[50.,0.],[50.,50.],[0.,50.]], device=device, dtype=dtype)
                ones = torch.ones((4,1), device=device, dtype=dtype)
                wc = (H @ torch.cat([corn,ones],1).t()).t()
                wc = wc[:,:2] / (wc[:,2:3] + 1e-8)

                span = wc.max(0).values - wc.min(0).values
                if span[0] > 500 or span[1] > 500:
                    continue

                if H is None:
                    continue

                count = int(inliers.sum().item())
                inlier_ratio = count / float(idx_a.shape[0] + 1e-8)

                if count >= 6 and inlier_ratio >= 0.15:
                    best_pair_h = H
                    best_pair_count = count
                    break

            if best_pair_h is None:
                continue

            overlap[i, j] = 1
            overlap[j, i] = 1

            pair_h[(i, j)] = best_pair_h
            pair_h[(j, i)] = torch.linalg.inv(best_pair_h)

            pair_score[(i, j)] = best_pair_count
            pair_score[(j, i)] = best_pair_count

    visited = [False] * n
    groups = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)

            for v in range(n):
                if overlap[u, v].item() == 1 and not visited[v]:
                    visited[v] = True
                    stack.append(v)

        groups.append(comp)

    groups.sort(key=len, reverse=True)
    main_group = groups[0]

    if len(main_group) == 1:
        out = img_list[main_group[0]].clamp(0, 1) * 255.0
        return out.byte(), overlap.cpu()

    anchor = main_group[0]
    best_deg = -1

    for i in main_group:
        deg = sum(1 for j in main_group if overlap[i, j].item() == 1)
        if deg > best_deg:
            best_deg = deg
            anchor = i

    img_to_anchor = {anchor: torch.eye(3, device=device, dtype=dtype)}
    done = {anchor}

    while len(done) < len(main_group):
        best_u = None
        best_v = None
        best_sc = -1

        for u in done:
            for v in main_group:
                if v in done:
                    continue
                if (u, v) in pair_h and pair_score[(u, v)] > best_sc:
                    best_sc = pair_score[(u, v)]
                    best_u = u
                    best_v = v

        if best_u is None:
            break

        img_to_anchor[best_v] = img_to_anchor[best_u] @ torch.linalg.inv(pair_h[(best_u, best_v)])
        done.add(best_v)

    used_imgs = [img_list[i] for i in main_group if i in img_to_anchor]
    used_h = [img_to_anchor[i] for i in main_group if i in img_to_anchor]

    all_corners = []

    for image, H in zip(used_imgs, used_h):
        h_img, w_img = image.shape[1], image.shape[2]

        corners = torch.tensor(
            [[0., 0.], [w_img - 1., 0.], [w_img - 1., h_img - 1.], [0., h_img - 1.]],
            device=device,
            dtype=dtype
        )
        ones = torch.ones((4, 1), device=device, dtype=dtype)

        warped = (H @ torch.cat([corners, ones], 1).t()).t()
        warped = warped[:, :2] / (warped[:, 2:3] + 1e-8)
        all_corners.append(warped)

    all_corners = torch.cat(all_corners, 0)

    min_x = torch.floor(all_corners[:, 0].min())
    min_y = torch.floor(all_corners[:, 1].min())
    max_x = torch.ceil(all_corners[:, 0].max())
    max_y = torch.ceil(all_corners[:, 1].max())

    out_w = int((max_x - min_x + 1).item())
    out_h = int((max_y - min_y + 1).item())

    shift = torch.eye(3, device=device, dtype=dtype)
    shift[0, 2] = -float(min_x.item())
    shift[1, 2] = -float(min_y.item())

    used_h = [shift @ H for H in used_h]

    warped_imgs = []
    warped_masks = []

    for image, H in zip(used_imgs, used_h):
        warped_img = K.geometry.transform.warp_perspective(
            image.unsqueeze(0), H.unsqueeze(0), dsize=(out_h, out_w),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        ones_mask = torch.ones((1, 1, image.shape[1], image.shape[2]), device=device, dtype=dtype)
        warped_mask = K.geometry.transform.warp_perspective(
            ones_mask, H.unsqueeze(0), dsize=(out_h, out_w),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        warped_mask = (warped_mask > 0.5).float()
        warped_imgs.append(warped_img)
        warped_masks.append(warped_mask)

    def get_weight(mask):
        weight = mask.clone()
        for _ in range(8):
            weight = K.filters.box_blur(weight, (31, 31))
            weight = weight * mask

        max_val = weight.max()
        if max_val > 1e-8:
            weight = weight / max_val

        return weight

    weight_sum = torch.zeros_like(warped_masks[0])
    weights = []

    for warped_mask in warped_masks:
        wt = get_weight(warped_mask)
        weights.append(wt)
        weight_sum = weight_sum + wt

    total = torch.zeros_like(warped_imgs[0])
    for warped_img, wt in zip(warped_imgs, weights):
        total = total + warped_img * wt

    pano = total / (weight_sum + 1e-8)

    simple_total = torch.zeros_like(warped_imgs[0])
    simple_count = torch.zeros_like(warped_masks[0])

    for warped_img, warped_mask in zip(warped_imgs, warped_masks):
        simple_total = simple_total + warped_img * warped_mask
        simple_count = simple_count + warped_mask

    simple_pano = simple_total / (simple_count + 1e-8)

    covered = (simple_count > 0).float()
    empty = (weight_sum < 1e-6).float() * covered
    pano = pano * (1.0 - empty) + simple_pano * empty

    pano = pano.squeeze(0).clamp(0.0, 1.0) * 255.0
    out_img = pano.byte()

    return out_img, overlap.cpu()