from stage1_listen.noise_suppression import clean_audio
from stage2_focus.enrollment import extract_voiceprint
from stage2_focus.speaker_isolation import isolate_target_speaker
from stage3_detect.keyword_spotting import detect_custom_word

def main():

    keyword = input("Enter keyword: ")

    print("\n===== STAGE 1 =====")
    clean_audio()

    print("\n===== STAGE 2A =====")
    extract_voiceprint()

    print("\n===== STAGE 2B =====")
    isolate_target_speaker()

    print("\n===== STAGE 3 =====")
    detect_custom_word(keyword)

if __name__ == "__main__":
    main()