import dateutil.parser


def build_response(filename, bounding_box, masks, is_located=None):
    """
    Prepares the data to be sent in a dictionary understandable by the requester.
    """

    if is_located is False:
        image = {}
    else:
        image = {
            "bounding_box": {
                "top": bounding_box["top"],
                "bottom": bounding_box["bottom"],
                "left": bounding_box["left"],
                "right": bounding_box["right"],
            },
            "masks": [
                {
                    "url": filename,
                    "level": mask[mask.rfind("_") + 1 : mask.rfind(".")],  # noqa
                }
                for mask in masks
            ],
        }

    return {"is_located": is_located if is_located is not None else "", "image": image}


def read_search_data(json_data):
    """
    Reads a request for satellite image search.
    """
    start_date = dateutil.parser.parse(json_data["start_date"])
    end_date = dateutil.parser.parse(json_data["end_date"])

    return start_date, end_date, json_data["bounding_box"]
