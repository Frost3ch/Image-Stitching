import numpy as np
import cv2
import glob
import os

def pre_warp(image, angle_deg, canvas_dims):
    """
    Apply a pre-warp transformation to simulate correction for a tilted view.
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    f = max(h, w)  # Focal length proportional to image size

    # Debug: Print input dimensions and parameters
    print(f"Input image dimensions: {h}x{w}")
    print(f"Tilt angle (degrees): {angle_deg}")
    print(f"Tilt angle (radians): {angle_rad}")
    print(f"Focal length: {f}")

    # Perspective transformation matrix
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]])
    tilt_matrix = np.array([[1, 0, 0],
                           [0, np.cos(angle_rad), -np.sin(angle_rad)],
                           [0, np.sin(angle_rad), np.cos(angle_rad)]])
    H = K @ tilt_matrix @ np.linalg.inv(K)

    # Normalize the homography matrix
    H = H / H[2, 2]

    # Shrink the prewarped image based on the tilt angle
    scaling_factor = np.cos(angle_rad)
    shrink_scale = np.array([
        [scaling_factor, 0, 0],
        [0, scaling_factor, 0],
        [0, 0, 1]
    ])
    H = shrink_scale @ H
    
    # Calculate destination corners to determine if image is being moved out of bounds
    h_canvas, w_canvas = canvas_dims[1], canvas_dims[0]
    corners = np.array([
        [0, 0, 1],
        [w-1, 0, 1],
        [0, h-1, 1],
        [w-1, h-1, 1]
    ]).T
    
    # Transform corners
    corners_transformed = H @ corners
    corners_transformed = corners_transformed / corners_transformed[2, :]
    
    # Get min and max x, y coordinates 
    min_x, max_x = np.min(corners_transformed[0, :]), np.max(corners_transformed[0, :])
    min_y, max_y = np.min(corners_transformed[1, :]), np.max(corners_transformed[1, :])
    
    # Calculate offset to keep image in view
    offset_x = max(0, -min_x) + min(w_canvas - max_x, 0) + w_canvas/4
    offset_y = max(0, -min_y) + min(h_canvas - max_y, 0) + h_canvas/4
    
    # Apply offset
    translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])
    H = translation @ H

    # Debug: Print the homography matrix
    print("Homography matrix (pre-warp):")
    print(H)

    # Warp the image
    warped_img = cv2.warpPerspective(image, H, canvas_dims)

    # Debug: Check for black output
    if np.all(warped_img == 0):
        print("Warning: Pre-warped image is completely black!")
    else:
        print("Pre-warped image has content!")
        # Count non-black pixels

    return warped_img

def main():
    # Load images
    images = []
    os.chdir("/Users/alpc/Desktop/Coding P/Image Stitching")
    sets = 1
    image_paths = []
    image_paths.append(sorted(glob.glob(f'./unstitchedImages/set2/*.jpg')))
    # for i in range(sets):
    #     image_paths.append(sorted(glob.glob(f'./unstitchedImages/set{sets-i}/*.jpg')))
    print(image_paths)
    
    # Load reference image (first image)
    ref_image_path = './unstitchedImages/ref2.jpg'
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        print(f"Error: Could not read reference image at {ref_image_path}")
        return
    ref_h, ref_w, _ = ref_image.shape

    
    print(f"Reference image: {ref_image_path}, dimensions: {ref_h}x{ref_w}")
    

    # # Load the rest of the images
    # for imageSet in image_paths:
    #     imgs = []
    #     for image_path in imageSet:
    #         img = cv2.imread(image_path)
    #         if img is None:
    #             print(f"Error: Could not read image at {image_path}")
    #             continue
    #         print(f"Loaded image: {image_path}, dimensions: {img.shape[0]}x{img.shape[1]}")
    #         imgs.append(img)
    #     images.append(imgs)

    # Create a larger canvas for stitching - make it significantly taller
    canvas_w = ref_w
    canvas_h = ref_h  # Increased vertical size to avoid cutoff
    stitched_map = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Place the reference image in centre of canvas
    center_x = canvas_w // 2
    center_y = canvas_h // 2  # Position it one reference image height from the top
    top_left_y, top_left_x = center_y - ref_h //2, center_x - ref_w // 2
    stitched_map[top_left_y:top_left_y + ref_h, top_left_x:top_left_x + ref_w] = ref_image
    
    # Save and display the initial map with reference image
    cv2.imwrite("reference_image_on_canvas.jpg", stitched_map)

    # Feature detector
    sift = cv2.SIFT_create()

    # Detect features for the reference image
    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_image, None)
    print(f'Detected {len(keypoints_ref)} reference features')

    # Process each image
    for setID,imgSet in enumerate(image_paths):
        tracker = 0
        for idx, img in enumerate(imgSet):
            img = cv2.imread(img)

            tracker+=1
            if tracker==1:
                warpAngle = int(input('enter warp angle: '))

            print(f"\n--- Processing Image {setID}_{idx+1} ---")
            
            # Pre-warp step remains the same

            warped_pre_img = pre_warp(img, warpAngle, (canvas_w, canvas_h))
            cv2.imwrite(f"pre_warp_{setID}_{idx+1}.jpg", warped_pre_img)

            # Detect features in the pre-warped image
            keypoints_img, descriptors_img = sift.detectAndCompute(warped_pre_img, None)
            print(f"Detected {len(keypoints_img) if keypoints_img else 0} features in image {idx+1}")

            # Skip if no descriptors are found
            if descriptors_ref is None or descriptors_img is None:
                print(f"No descriptors found for Image {setID}_{idx+1}. Skipping...")
                continue

            # Convert descriptors to np.float32 for FLANN
            descriptors_ref = np.asarray(descriptors_ref, dtype=np.float32)
            descriptors_img = np.asarray(descriptors_img, dtype=np.float32)

            # Use FLANN-based matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(descriptors_img, descriptors_ref, k=2)  # Note the order
            print(f"Found {len(matches)} initial matches")

            # Apply Lowe's ratio test - try with a more permissive threshold

            all_matches = []
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:  # More permissive ratio
                    good_matches.append(m)
                all_matches.append(m)
            
            #match_img = cv2.drawMatches(warped_pre_img, keypoints_img, stitched_map, keypoints_ref,all_matches[:400], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #cv2.imwrite(f"matches_{setID}_{idx+1}.jpg", match_img)
            
            print(f"Found {len(good_matches)} good matches after ratio test")

            # Need at least 8 good matches to proceed (reduced from 10)
            if len(good_matches) < 12:
                print(f"Insufficient matches for Image {setID}_{idx+1}. Skipping...")
                continue
        
            src_pts = np.float32([keypoints_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
            # Calculate homography from pre-warped image to reference
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, 2.0)
            
            if H is None:
                print(f"Failed to calculate homography for Image {setID}_{idx+1}. Skipping...")
                continue

            # Debug: Print and inspect homography matrix
            print(f"Homography for Image {setID}_{idx+1}:")
            print(H)
            
            # Count inliers from RANSAC
            inliers = np.sum(mask)
            print(f"RANSAC inliers: {inliers} out of {len(good_matches)}")
            
            # CRITICAL FIX: Check if homography is too extreme
            h_det = np.linalg.det(H[:2, :2])
            if abs(h_det) < 0.01 or abs(h_det) > 10:
                print(f"WARNING: Homography determinant is extreme: {h_det}. Oh well...")

        
            # Calculate translation to position reference image on canvas
            # Ensure it's float32 to avoid type errors in warpPerspective
            if idx==0:
                ref_to_canvas = np.array([
                    [1.0, 0.0, float(top_left_x)], 
                    [0.0, 1.0, float(top_left_y)], 
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
                # Combine matrices to get final transformation (ensure float32 type)
                final_H = ref_to_canvas @ H.astype(np.float32)
            else:
                final_H = H

            # IMPORTANT: Check if any values in the homography are NaN or infinite
            if np.any(np.isnan(final_H)) or np.any(np.isinf(final_H)):
                print(f"ERROR: Homography contains NaN or infinite values. Skipping image {setID}_{idx+1}.")
                continue
            
            
            # Homography warp :-)
            try:
                warped_img = cv2.warpPerspective(warped_pre_img, final_H, (canvas_w, canvas_h))
                non_black1 = np.sum(warped_img > 0)
                print(f"Method 1 non-black pixels: {non_black1}")
            except Exception as e:
                print(f"Error in method 1: {e}")
                non_black1 = 0
                warped_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            
            
            # Save the warped image for debugging
            #cv2.imwrite(f"warped_{idx+1}.jpg", warped_img)x
            
            # Create alpha masks for blending
            alpha = 0.5  # Equal blending weight

            mask = (warped_img > 0).astype(np.uint8)
            existing_mask = (stitched_map > 0).astype(np.uint8)
            overlap_mask = mask & existing_mask
                        
            #stitched_map[top_left_y:top_left_y + ref_h, top_left_x:top_left_x + ref_w] = ref_image
            # Blend in overlap regions, otherwise just use the non-zero pixels
            stitched_map = np.where(overlap_mask, alpha * warped_img.astype(np.float32) + 
                                (1 - alpha) * stitched_map.astype(np.float32), stitched_map).astype(np.uint8)
            stitched_map = np.where(mask & ~overlap_mask, warped_img, stitched_map)
            
            # Save intermediate result
            cv2.imwrite(f"stitched_intermediate_{setID}_{idx+1}.jpg", stitched_map)

            gray_map = cv2.cvtColor(stitched_map, cv2.COLOR_BGR2GRAY)
            maskE = cv2.inRange(gray_map, 1, 255)
            keypoints_ref, descriptors_ref = sift.detectAndCompute(stitched_map, maskE)

        print('Completed stitching process')

        # Save and display the final stitched map
        #stitched_map[top_left_y:top_left_y + ref_h, top_left_x:top_left_x + ref_w] = ref_image
        cv2.imwrite("stitched_map_final.jpg", stitched_map)
        cv2.imshow("Final Stitched Image", stitched_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()