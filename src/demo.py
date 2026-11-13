
from pathlib import Path

from src.inference import load_artifacts, run_transaction


BASE_DIR = Path(__file__).resolve().parents[1]

MEMBERS = {
    "1": "Yohan",
    "2": "Janviere",
    "3": "Josue",
    "4": "Thomas",
}


def select_member():
    """Allow the user to select an authorized group member."""
    print("\nSelect the authorized member:")

    for key, name in MEMBERS.items():
        print(f"{key}. {name}")

    while True:
        choice = input("\nChoose a member (1-4): ").strip()

        if choice in MEMBERS:
            return MEMBERS[choice]

        print("Invalid choice. Please choose 1, 2, 3 or 4.")


def get_member_files(member):
    """Get the face and voice sample paths for a member."""
    image_path = BASE_DIR / "data" / "images" / member / f"{member}_neutral.jpg"
    audio_path = BASE_DIR / "data" / "audio" / member / f"{member}_recording1.m4a"

    return image_path, audio_path


def main():
    print("\n===================================")
    print("   MULTIMODAL AUTHENTICATION DEMO")
    print("===================================")

    artifacts = load_artifacts()

    while True:
        print("\nSelect a scenario:")
        print("1. Successful transaction")
        print("2. Unauthorized face attempt")
        print("3. Unauthorized voice attempt")
        print("4. Exit")

        choice = input("\nChoose a scenario (1-4): ").strip()

        if choice == "4":
            print("Exiting demo. Goodbye!")
            break

        if choice not in ("1", "2", "3"):
            print("Invalid choice. Please choose 1, 2, 3 or 4.")
            continue

        # Select the expected authorized member
        member = select_member()

        image_path, audio_path = get_member_files(member)

        if choice == "2":
            # Use an unknown face and the selected member's voice
            image_path = BASE_DIR / "data" / "images" / "unauthorized" / "unknown.jpg"

        elif choice == "3":
            # Use the selected member's face and another member's voice
            other_members = [
                name for name in MEMBERS.values() if name != member
            ]

            print("\nSelect a different speaker:")
            for index, name in enumerate(other_members, start=1):
                print(f"{index}. {name}")

            while True:
                speaker_choice = input("\nChoose a speaker: ").strip()

                if speaker_choice.isdigit():
                    index = int(speaker_choice) - 1

                    if 0 <= index < len(other_members):
                        speaker = other_members[index]
                        break

                print("Invalid choice. Please select a listed speaker.")

            audio_path = (
                BASE_DIR / "data" / "audio" / speaker
                / f"{speaker}_recording1.m4a"
            )

        print(f"\n--- Scenario: {choice} ---")
        print(f"Expected member: {member}")

        if choice == "3":
            print(f"Voice sample belongs to: {speaker}")

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        if not audio_path.exists():
            print(f"Audio not found: {audio_path}")
            continue

        try:
            result = run_transaction(
                str(image_path),
                str(audio_path),
                artifacts
            )

            print("\nFACE VERIFICATION")
            print(f"Recognized member: {result['face']['member']}")
            print(f"Confidence: {result['face']['confidence']:.2%}")

            if "voice" in result:
                print("\nVOICE VERIFICATION")
                print(f"Recognized speaker: {result['voice']['speaker']}")
                print(f"Confidence: {result['voice']['confidence']:.2%}")

            print("\nTRANSACTION RESULT")

            if result["status"] == "APPROVED":
                print("ACCESS APPROVED")
                print(f"User: {result['face']['member']}")
                print(f"Recommended product: {result['product']['category']}")
                print(
                    "Prediction confidence: "
                    f"{result['product']['confidence']:.2%}"
                )
            else:
                print("ACCESS DENIED")
                print(f"Reason: {result['reason']}")

        except Exception as error:
            print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
