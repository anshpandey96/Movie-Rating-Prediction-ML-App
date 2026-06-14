const numericInputs = document.querySelectorAll('input[type="number"]');

numericInputs.forEach((input) => {
    input.addEventListener("input", () => {
        const min = Number(input.min || 0);
        if (Number(input.value) < min) {
            input.value = min;
        }
    });
});
