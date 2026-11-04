"""Export AVTR latent queries and routing markers to Hugging Face format."""


def main() -> None:
    from .bake import main as export_main

    export_main()


if __name__ == "__main__":
    main()
