import sys
from PyQt5.QtWidgets import QApplication
from Modules.TouchAwareModule import TouchDetectionApp


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = TouchDetectionApp()
    window.show()
    sys.exit(app.exec_())