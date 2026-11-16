def get_center_of_bbox(bbox):
    # Log the coordinates before calculating the box centre.
    print(f"[get_center_of_bbox] Calculating centre for bbox={bbox}")
    x1,y1,x2,y2 = bbox

    # Store the result so it can be reported before it is returned.
    center = int((x1+x2)/2),int((y1+y2)/2)
    print(f"[get_center_of_bbox] Calculated centre={center}")
    return center

def get_bbox_width(bbox):
    # Log the coordinates before calculating the horizontal box width.
    print(f"[get_bbox_width] Calculating width for bbox={bbox}")
    width = bbox[2]-bbox[0]
    print(f"[get_bbox_width] Calculated width={width}")
    return width
