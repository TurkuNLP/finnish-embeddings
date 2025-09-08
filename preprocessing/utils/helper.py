def print_announcement(announcement:str):
    announcement_len = len(announcement) + 2
    print(f"┌{'─'* announcement_len}┐")
    print(f"│ {announcement} │")
    print(f"└{'─'* announcement_len}┘")

def update_article_count(num_articles: int, new_shape: int, message: str):
    deleted = num_articles - new_shape
    print(f"- {message} -> {new_shape:,} ({deleted:,} deleted)")
    return new_shape