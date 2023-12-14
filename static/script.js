// script.js

//display rating star
function rating_star(rating) {
    var starsHTML = '';
    var filledStars = Math.floor(rating);
    var outlinedStars = 5 - filledStars;

    for (var i = 0; i < filledStars; i++) {
        starsHTML += '<span style="color: #fcba03;">★</span>';  // Filled star character
    }

    for (var i = 0; i < outlinedStars; i++) {
        starsHTML += '<span style="color: #fcba03;">☆</span>';  // Outlined star character
    }

    return starsHTML;
}

// like button
function toggleLike(likeContainer, restaurant, cuisineType, priceLevel) {
    var heartSymbol = likeContainer.querySelector('#heart-symbol');

    // Assuming you have a temporary storage array
    var likedRestaurants = JSON.parse(localStorage.getItem('likedRestaurants')) || [];

    // Check if the restaurant is not already in the liked list
    var isLiked = likedRestaurants.some(function (liked) {
        return liked.restaurant === restaurant && liked.cuisineType === cuisineType && liked.priceLevel === priceLevel;
    });

    if (isLiked) {
        // Remove the restaurant from the liked list
        likedRestaurants = likedRestaurants.filter(function (liked) {
            return !(liked.restaurant === restaurant && liked.cuisineType === cuisineType && liked.priceLevel === priceLevel);
        });

        // Update the temporary storage
        localStorage.setItem('likedRestaurants', JSON.stringify(likedRestaurants));

        heartSymbol.innerHTML = '&#x2661; Like';

        // Provide user feedback (you can customize this part)
        alert('Restaurant Unliked!');
    } else {
        // Add the restaurant to the liked list
        likedRestaurants.push({
            restaurant: restaurant,
            cuisineType: cuisineType,
            priceLevel: priceLevel
        });

        // Update the temporary storage
        localStorage.setItem('likedRestaurants', JSON.stringify(likedRestaurants));

        heartSymbol.innerHTML = '&#x2665; Liked';

        // Provide user feedback (you can customize this part)
        alert('Restaurant Liked!');
    }

    // Display liked restaurants
    displayLikedRestaurants();
}

function displayLikedRestaurants() {
    var likedRestaurants = JSON.parse(localStorage.getItem('likedRestaurants')) || [];
    var likedRestaurantsList = document.getElementById('liked-restaurants-list');

    // Clear previous content
    likedRestaurantsList.innerHTML = '';

    // Display each liked restaurant
    likedRestaurants.forEach(function (liked) {
        var listItem = document.createElement('li');

        // Create separate elements for each part of the information
        var restaurantName = document.createElement('div');
        var cuisineType = document.createElement('div');
        var priceLevel = document.createElement('div');

        // Set the content for each element
        restaurantName.textContent = liked.restaurant;
        cuisineType.textContent = `- ${liked.cuisineType}`;
        priceLevel.textContent = `- PRICE LEVEL: ${liked.priceLevel}`;

        cuisineType.style.paddingLeft = '40px';
        priceLevel.style.paddingLeft = '40px';
        priceLevel.style.paddingBottom = '10px';

        // Append the elements to the list item
        listItem.appendChild(restaurantName);
        listItem.appendChild(cuisineType);
        listItem.appendChild(priceLevel);

        // Append the list item to the likedRestaurantsList
        likedRestaurantsList.appendChild(listItem);
    });

    // Display the section
    var likedRestaurantsSection = document.getElementById('liked-restaurants-section');
    likedRestaurantsSection.style.display = 'block';
}

window.addEventListener('load', function () {
    displayLikedRestaurants();
});

function clearLikedItems() {
    // Clear the liked items from localStorage
    localStorage.removeItem('likedRestaurants');

    // Clear the liked restaurants list in the HTML
    var likedRestaurantsList = document.getElementById('liked-restaurants-list');
    likedRestaurantsList.innerHTML = '';
}

// Function to open the modal
function openModal() {
    var modal = document.getElementById('myModal');
    modal.style.display = 'flex';  // Set to 'flex' to center the content vertically
}

function closeModal() {
    var modal = document.getElementById('myModal');
    modal.style.display = 'none';
}

// Close the modal if the user clicks outside of it
window.onclick = function (event) {
    var modal = document.getElementById('myModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

/* predict2.html */
function populateLikedRestaurants() {
    var likedRestaurants = JSON.parse(localStorage.getItem('likedRestaurants')) || [];
    var likedRestaurantsContainer = document.getElementById('likedRestaurantsContainer');
    console.log('hi')

    // Clear previous content
    likedRestaurantsContainer.innerHTML = '';

    // Create input fields for each liked restaurant parameter
    likedRestaurants.forEach(function (liked) {
        var inputRestaurant = document.createElement('input');
        inputRestaurant.type = 'hidden';
        inputRestaurant.name = 'restaurant';
        inputRestaurant.value = liked.restaurant;

        var inputCuisineType = document.createElement('input');
        inputCuisineType.type = 'hidden';
        inputCuisineType.name = 'cuisineType';
        inputCuisineType.value = liked.cuisineType;

        var inputPriceLevel = document.createElement('input');
        inputPriceLevel.type = 'hidden';
        inputPriceLevel.name = 'priceLevel';
        inputPriceLevel.value = liked.priceLevel;

        // Append the input fields to the container
        likedRestaurantsContainer.appendChild(inputRestaurant);
        likedRestaurantsContainer.appendChild(inputCuisineType);
        likedRestaurantsContainer.appendChild(inputPriceLevel);
        console.log('hihi')
        console.log(likedRestaurantsContainer)
    });
}

// Call the function when the page is loaded
window.addEventListener('load', function () {
    populateLikedRestaurants();
});








