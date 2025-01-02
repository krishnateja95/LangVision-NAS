from PIL import Image, ImageDraw, ImageFont

def create_image_from_dict(data):
    # Create a blank image
    width = 100 * len(data)  # Adjust width based on number of elements
    height = 200
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Define box size and spacing
    box_size = 50
    spacing = 30
    font = ImageFont.load_default()

    # Calculate starting position
    x, y = spacing, height // 2 - box_size // 2

    # Draw boxes and arrows
    for i, value in enumerate(data):
        # Draw the box
        draw.rectangle([x, y, x + box_size, y + box_size], outline='black', width=2)
        
        # Add the number inside the box (centered)
        text_x = x + box_size // 2
        text_y = y + box_size // 2
        text_w, text_h = draw.textsize(str(value), font=font)
        draw.text((text_x - text_w // 2, text_y - text_h // 2), str(value), fill='black', font=font)

        # Draw an arrow to the next box (if not the last box)
        if i < len(data) - 1:
            arrow_start = (x + box_size, y + box_size // 2)
            arrow_end = (x + box_size + spacing, y + box_size // 2)
            draw.line([arrow_start, arrow_end], fill='black', width=2)
            # Draw arrowhead
            arrowhead = [
                (arrow_end[0] - 10, arrow_end[1] - 5),
                (arrow_end[0] - 10, arrow_end[1] + 5),
                arrow_end
            ]
            draw.polygon(arrowhead, fill='black')

        # Move to the next position
        x += box_size + spacing

    return img

# Example dictionary-like data structure
data = [2, 4, 8, 8, 4, 2]
image = create_image_from_dict(data)
image.show()
