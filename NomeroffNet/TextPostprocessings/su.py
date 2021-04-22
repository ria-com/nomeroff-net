from .xx_xx import xx_xx


class su(xx_xx):
    def find(self, text, strong=True):
        text = self.check_is_str(text)
        text = self.delete_all_black_list_characters(text)
        text = text.upper()

        if len(text) < len(self.STANDART):
            return text

        if len(self.STANDART):
            match = self.findFully(text)
            if match:
                return match.group(0)

        if not strong:
            return self.findSimilary(text)
        return text
